# FlashVSR Architecture Diagram

## 🏗️ **FlashVSR Model Architecture**

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              FlashVSR Pipeline                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Input Video (LQ) ──┐                                                          │
│                     │                                                          │
│                     ▼                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                    Three-Stage Distillation Pipeline                    │   │
│  │                                                                         │   │
│  │  Stage 1: LQ Projection (Buffer_LQ4x_Proj)                             │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │   │
│  │  │  Input: (B, 3, F, H, W) → Output: Multi-scale LQ latents        │   │   │
│  │  │  • 3D Convolution layers                                         │   │   │
│  │  │  • Stream processing (2-7 frames at a time)                     │   │   │
│  │  │  • Output: [LQ_latents[0], LQ_latents[1], ..., LQ_latents[N]]   │   │   │
│  │  └─────────────────────────────────────────────────────────────────┘   │   │
│  │                                                                         │   │
│  │  Stage 2: Diffusion Transformer (WanModel)                             │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │   │
│  │  │  • Locality-Constrained Sparse Attention (LCSA)                 │   │   │
│  │  │  • Block-Sparse Attention with dynamic masking                  │   │   │
│  │  │  • RoPE (Rotary Position Embedding)                            │   │   │
│  │  │  • KV Cache for streaming efficiency                           │   │   │
│  │  │  • 30 transformer blocks                                       │   │   │
│  │  └─────────────────────────────────────────────────────────────────┘   │   │
│  │                                                                         │   │
│  │  Stage 3: Tiny Conditional Decoder (TCDecoder)                         │   │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │   │
│  │  │  • Multi-scale channels: [512, 256, 128, 128]                  │   │   │
│  │  │  • Latent channels: 16 + 768 = 784                             │   │   │
│  │  │  • Efficient reconstruction without VAE                         │   │   │
│  │  │  • Parallel processing capability                               │   │   │
│  │  └─────────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│                     ▼                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                    Color Correction (Optional)                          │   │   │
│  │  • Wavelet-based color matching                                        │   │   │
│  │  • ADAIN (Adaptive Instance Normalization)                             │   │   │
│  │  • Chunked processing for memory efficiency                            │   │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│                     ▼                                                          │
│  Output Video (HQ) ──┘                                                         │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 🔧 **Key Components Breakdown**

### **1. LQ Projection (Buffer_LQ4x_Proj)**
```
Input: (B, 3, F, H, W)  →  Output: Multi-scale LQ latents
├── 3D Convolution layers
├── Stream processing (2-7 frames chunks)
├── Output: [LQ_latents[0], LQ_latents[1], ..., LQ_latents[N]]
└── Memory: ~2-4GB VRAM per frame
```

### **2. Diffusion Transformer (WanModel)**
```
Architecture: 30 transformer blocks
├── Locality-Constrained Sparse Attention (LCSA)
│   ├── Block-Sparse Attention with dynamic masking
│   ├── Top-k selection for efficiency
│   └── Local range: 9-11 (configurable)
├── RoPE (Rotary Position Embedding)
│   ├── 3D positional encoding (time, height, width)
│   └── Streaming-friendly design
├── KV Cache System
│   ├── pre_cache_k, pre_cache_v for streaming
│   ├── Memory-efficient attention
│   └── Bridge train-test resolution gap
└── Memory: ~8-16GB VRAM (depends on resolution)
```

### **3. Tiny Conditional Decoder (TCDecoder)**
```
Multi-scale Architecture:
├── Channels: [512, 256, 128, 128]
├── Latent channels: 16 + 768 = 784
├── Efficient reconstruction (no VAE needed)
├── Parallel processing capability
└── Memory: ~4-8GB VRAM
```

## 🚀 **FlashVSR vs FlashVSR-Tiny Comparison**

| Component | FlashVSR (Full) | FlashVSR-Tiny | Memory Savings |
|-----------|-----------------|---------------|----------------|
| **DiT Model** | 14B parameters | 1.3B parameters | ~10x smaller |
| **VAE** | Full VAE decoder | TCDecoder only | ~5x smaller |
| **Attention** | Dense + Sparse | Sparse only | ~3x faster |
| **Total VRAM** | 24-40GB | 8-16GB | ~2-3x less |
| **Speed** | ~17 FPS | ~25-30 FPS | ~1.5x faster |

## 📊 **Memory Requirements by Resolution**

| Resolution | Full Model | Tiny Model | Vidaio Compatible |
|------------|------------|------------|-------------------|
| **480p→1080p** | 16-24GB | 6-10GB | ✅ **YES** |
| **1080p→4K** | 24-32GB | 8-12GB | ✅ **YES** |
| **480p→4K** | 20-28GB | 8-12GB | ✅ **YES** |
| **4K→8K** | 32-40GB | 12-16GB | ✅ **YES** |

## ⚡ **Streaming Processing Flow**

```
1. Input Video (5-10 seconds, 150-300 frames)
   ↓
2. Frame Chunking (2-7 frames per chunk)
   ↓
3. LQ Projection (per chunk)
   ↓
4. Diffusion Transformer (streaming with KV cache)
   ↓
5. TCDecoder (efficient reconstruction)
   ↓
6. Color Correction (optional)
   ↓
7. Output Video (full duration maintained)
```

## 🎯 **Vidaio Subnet Compatibility**

**FlashVSR-Tiny is PERFECT for Vidaio subnet**:

✅ **Memory**: 8-16GB VRAM (fits A100-40GB easily)  
✅ **Speed**: 25-30 FPS (faster than RealESRGAN)  
✅ **Quality**: State-of-the-art diffusion-based upscaling  
✅ **Duration**: Handles full 5-10 second videos  
✅ **Resolution**: Supports all Vidaio task types  

**FlashVSR-Tiny > RealESRGAN for Vidaio mining!**


