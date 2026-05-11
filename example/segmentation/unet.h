// SMP-Unet (ResNet34 encoder + plain decoder) for binary segmentation.
//
// Matches segmentation_models_pytorch.Unet(encoder='resnet34',
// decoder_channels=[256,128,64,32,16], classes=1). State_dict keys map 1:1
// to PyTorch's; load_safetensors() at the bottom copies in named tensors.
//
// This is the "after" pass using the new high-level cg::nn API. Everything
// here is PyTorch-shaped: each layer is a named Module member, forward()
// chains them by passing nn::Tensor through. Compare against the previous
// ~280-line version that threaded ComputeGraph& + shape vectors manually.

#pragma once

#include "cg.h"
#include "nn.h"
#include "safetensors.h"

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace seg {

using cg::ComputeGraph;
using cg::Node;
using cg::InputNode;
using cg::nn::Tensor;

// ----- ResNet34 BasicBlock --------------------------------------------------
struct BasicBlock : public cg::nn::Module {
    bool has_downsample;
    cg::nn::Conv2D      conv1, conv2;
    cg::nn::BatchNorm2D bn1,   bn2;
    cg::nn::Conv2D      ds_conv;          // 1x1 stride-s shortcut, only when needed
    cg::nn::BatchNorm2D ds_bn;

    BasicBlock(int Cin, int Cout, int stride)
        : has_downsample(stride != 1 || Cin != Cout),
          conv1(Cin,  Cout, 3, stride, 1, /*bias=*/false),
          conv2(Cout, Cout, 3, 1,      1, /*bias=*/false),
          bn1(Cout), bn2(Cout),
          ds_conv(Cin, Cout, 1, stride, 0, /*bias=*/false),
          ds_bn(Cout) {}

    Tensor forward(Tensor x) override {
        auto y  = bn2(conv2(cg::nn::relu(bn1(conv1(x)))));
        auto sc = has_downsample ? ds_bn(ds_conv(x)) : x;
        // y += sc
        auto* g = y.graph();
        auto* sum = g->emplace<cg::MatAddNode>(y.node(), sc.node());
        return cg::nn::relu(Tensor(g, sum, y.shape()));
    }
};

// ----- SMP UnetDecoderBlock: upsample(2) -> cat(skip) -> double conv --------
struct DecoderBlock : public cg::nn::Module {
    bool has_skip;
    int  Cin_after_cat, Cout;
    cg::nn::Conv2D      conv1, conv2;
    cg::nn::BatchNorm2D bn1,   bn2;

    DecoderBlock(int Cin_after_cat, int Cout, bool has_skip)
        : has_skip(has_skip),
          Cin_after_cat(Cin_after_cat), Cout(Cout),
          conv1(Cin_after_cat, Cout, 3, 1, 1, /*bias=*/false),
          conv2(Cout,          Cout, 3, 1, 1, /*bias=*/false),
          bn1(Cout), bn2(Cout) {}

    // DecoderBlock's forward signature deviates from Module's (it needs a
    // skip Tensor as a 2nd arg). Override forward() to handle the no-skip
    // case via Module API; provide a 2-arg overload for blocks 0-3.
    Tensor forward(Tensor x) override {                        // no-skip variant
        if (has_skip) throw std::runtime_error("DecoderBlock: skip required");
        auto up = cg::nn::upsample_nearest(x, 2);
        return cg::nn::relu(bn2(conv2(cg::nn::relu(bn1(conv1(up))))));
    }
    Tensor forward(Tensor x, Tensor skip) {
        // SMP convention: torch.cat([upsampled, skip], dim=1).
        auto cat = cg::nn::concat(cg::nn::upsample_nearest(x, 2), skip);
        if (cat.size(1) != Cin_after_cat)
            throw std::runtime_error("DecoderBlock: in-channel mismatch");
        return cg::nn::relu(bn2(conv2(cg::nn::relu(bn1(conv1(cat))))));
    }
};

// ----- Full U-Net -----------------------------------------------------------
struct UNet : public cg::nn::Module {
    cg::nn::Conv2D      stem_conv{3, 64, 7, /*stride=*/2, /*pad=*/3, /*bias=*/false};
    cg::nn::BatchNorm2D stem_bn{64};
    std::vector<BasicBlock>   layer1, layer2, layer3, layer4;
    std::vector<DecoderBlock> dec;
    cg::nn::Conv2D      head{16, 1, 3, /*stride=*/1, /*pad=*/1, /*bias=*/true};

    UNet() {
        // Layout the encoder stages. Reserve before emplace so the BasicBlock
        // addresses are stable across the constructor (their member ptrs into
        // the graph are set lazily on first forward(), but vector reallocation
        // before that would still move the values).
        layer1.reserve(3); layer2.reserve(4); layer3.reserve(6); layer4.reserve(3); dec.reserve(5);
        for (int i = 0; i < 3; ++i) layer1.emplace_back(64,  64, 1);
        layer2.emplace_back(64, 128, 2);
        for (int i = 0; i < 3; ++i) layer2.emplace_back(128, 128, 1);
        layer3.emplace_back(128, 256, 2);
        for (int i = 0; i < 5; ++i) layer3.emplace_back(256, 256, 1);
        layer4.emplace_back(256, 512, 2);
        for (int i = 0; i < 2; ++i) layer4.emplace_back(512, 512, 1);

        dec.emplace_back(512 + 256, 256, true);   // block0
        dec.emplace_back(256 + 128, 128, true);   // block1
        dec.emplace_back(128 +  64,  64, true);   // block2
        dec.emplace_back( 64 +  64,  32, true);   // block3
        dec.emplace_back( 32,        16, false);  // block4 (no skip)
    }

    // Returns raw 1-channel logits, shape [N, 1, H, W]. H and W must be % 32.
    Tensor forward(Tensor x) override {
        if (x.size(2) % 32 != 0 || x.size(3) % 32 != 0)
            throw std::runtime_error("UNet: H and W must be divisible by 32");

        auto skip0 = cg::nn::relu(stem_bn(stem_conv(x)));        // 1/2,  64ch
        auto y     = cg::nn::maxpool2d(skip0, 3, 2, 1);          // 1/4,  64ch

        for (auto& bb : layer1) y = bb(y);  auto skip1 = y;      // 1/4,  64ch
        for (auto& bb : layer2) y = bb(y);  auto skip2 = y;      // 1/8, 128ch
        for (auto& bb : layer3) y = bb(y);  auto skip3 = y;      // 1/16,256ch
        for (auto& bb : layer4) y = bb(y);                       // 1/32,512ch

        y = dec[0].forward(y, skip3);
        y = dec[1].forward(y, skip2);
        y = dec[2].forward(y, skip1);
        y = dec[3].forward(y, skip0);
        y = dec[4].forward(y);                                   // no skip

        return head(y);                                          // [N, 1, H, W] logits
    }

    // ----- safetensors loading -------------------------------------------
    // PyTorch key layout (matches smp.Unet with resnet34 encoder):
    //   encoder.conv1.weight                            [64, 3, 7, 7]
    //   encoder.bn1.{weight,bias,running_mean,running_var}        [64]
    //   encoder.layerN.M.{conv1,conv2}.weight           [Cout, Cin, 3, 3]
    //   encoder.layerN.M.{bn1,bn2}.*                              [Cout]
    //   encoder.layerN.0.downsample.{0.weight, 1.*}               (when stride!=1 or chs differ)
    //   decoder.blocks.K.{conv1,conv2}.0.weight + .1.*
    //   segmentation_head.0.{weight, bias}              [1, 16, 3, 3] / [1]

    static void load_conv(const safetensors::File& f, const std::string& key, cg::nn::Conv2D& c) {
        safetensors::copy_into(f, key, c.weight());
    }
    static void load_bn(const safetensors::File& f, const std::string& base, cg::nn::BatchNorm2D& bn) {
        safetensors::copy_into(f, base + ".weight",       bn.weight());
        safetensors::copy_into(f, base + ".bias",         bn.bias());
        safetensors::copy_into(f, base + ".running_mean", bn.running_mean());
        safetensors::copy_into(f, base + ".running_var",  bn.running_var());
    }
    static void load_flat_bias(const safetensors::File& f, const std::string& key,
                               cg::Tensor& dst) {
        auto it = f.index.find(key);
        if (it == f.index.end()) throw std::runtime_error("safetensors: missing '" + key + "'");
        if (it->second.length_bytes != (size_t)dst.numel() * sizeof(float))
            throw std::runtime_error("safetensors: byte size mismatch for '" + key + "'");
        std::memcpy(dst.data(), f.data.data() + it->second.offset_bytes, it->second.length_bytes);
    }

    void load_safetensors(const safetensors::File& f) {
        load_conv(f, "encoder.conv1.weight", stem_conv);
        load_bn  (f, "encoder.bn1",          stem_bn);

        auto load_stage = [&](std::vector<BasicBlock>& layer, const std::string& name) {
            for (size_t i = 0; i < layer.size(); ++i) {
                auto base = "encoder." + name + "." + std::to_string(i);
                load_conv(f, base + ".conv1.weight", layer[i].conv1);
                load_bn  (f, base + ".bn1",          layer[i].bn1);
                load_conv(f, base + ".conv2.weight", layer[i].conv2);
                load_bn  (f, base + ".bn2",          layer[i].bn2);
                if (layer[i].has_downsample) {
                    load_conv(f, base + ".downsample.0.weight", layer[i].ds_conv);
                    load_bn  (f, base + ".downsample.1",        layer[i].ds_bn);
                }
            }
        };
        load_stage(layer1, "layer1");
        load_stage(layer2, "layer2");
        load_stage(layer3, "layer3");
        load_stage(layer4, "layer4");

        for (size_t i = 0; i < dec.size(); ++i) {
            auto base = "decoder.blocks." + std::to_string(i);
            load_conv(f, base + ".conv1.0.weight", dec[i].conv1);
            load_bn  (f, base + ".conv1.1",        dec[i].bn1);
            load_conv(f, base + ".conv2.0.weight", dec[i].conv2);
            load_bn  (f, base + ".conv2.1",        dec[i].bn2);
        }
        safetensors::copy_into(f, "segmentation_head.0.weight", head.weight());
        load_flat_bias(f, "segmentation_head.0.bias", head.bias());
    }
};

} // namespace seg
