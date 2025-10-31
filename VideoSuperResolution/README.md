---
license: unknown
---
You will find these models in your Edge browser directory:
C:\Users\{user}\AppData\Local\Microsoft\Edge\User Data\EdgeOnnxRuntimeDirectML\1.12.1.11

# Video super resolution in Microsoft Edge

Have you ever wished you could watch your favorite videos in high definition, even if they were originally recorded in lower quality? Well, now you can, thanks to a new feature we are experimenting with in Edge Canary: video super resolution (VSR).

Video super resolution uses machine learning to enhance the quality of video viewed in Microsoft Edge by using graphics card agnostic algorithms to remove blocky compression artifacts and upscale the video resolution, so you can enjoy crisp and clear videos on YouTube and other video streaming platforms without sacrificing bandwidth.

Due to the computing power required to upscale videos, video super resolution (VSR) is currently offered when the following conditions are met:

The device has one of the following graphics cards (GPUs): Nvidia RTX 20/30/40 series OR AMD RX5700-RX7800 series GPUs. [1]
The video is played at less than 720p resolution.
The device is not on battery power.
Both the height and width of the video are greater than 192 pixels.
The video is not protected with Digital Rights Management technologies like PlayReady or Widevine. Frames from these protected videos are not accessible to the browser for processing.
[1] Note: We are working on automatic Hybrid GPU support for laptops with multiple GPUs. Meanwhile, you can try VSR by changing Windows settings to force Edge to run on your discrete GPU.

Video super resolution is automatically enabled by Edge and indicated by an HD icon on the address. The feature can be computationally intensive, so this icon allows a user to be in full control of enabling or disabling the feature.

## Availability

As noted above, we’ve started experimenting with a small set of customers in the Canary channel and will continue to make this feature available to additional customers over the coming weeks. We are also looking forward to expanding the list of supported graphics cards in the future.

## Behind the Scenes

Let’s go into some additional details about how video super resolution, or VSR, works behind the scenes.

## ONNX Runtime and DirectML

VSR in Microsoft Edge builds on top of ONNX Runtime and DirectML making our solution portable across GPU vendors and allowing VSR to be available to more users. Additional graphics cards which support these technologies and have sufficient computing power will receive support in the future. The ONNX Runtime and DirectML teams have fine-tuned their technology over many years, resulting in VSR making the most of the performance and capabilities of your graphics card’s processing power. ONNX Runtime handles loading ML models packaged as an .onnx files and uses DirectML, which handles the optimization and evaluation of the ML workload by leveraging the available GPU capabilities such as the native ML tensor processing to achieve the maximum execution throughput at a high framerate.

## Storing Machine Learning Models

To preserve disk space, the components and models that VSR requires are only added to your device when we detect a compatible GPU. The presence of a component named “Edge Video Super Resolution” when visiting edge://components/ in Edge Canary is a signal that your GPU is supported by the video super resolution feature. This component-based approach lets us ship specific and multiple models based on device capability and performance.

## DirectX 11 Interop with DirectML

To support VSR, we have built a new DX12 presentation pipeline in Microsoft Edge. Chromium, which Microsoft Edge is built on, uses DX11 for video decode/rasterization and generates DX11 textures after video decode. DirectML on the other hand only works with DX12 buffers. To support VSR, we built a new flexible DX12 pipeline into the Chromium engine that’s embedded in Microsoft Edge. Our new pipeline runs the shaders to convert DX11 textures into DirectML buffers/tensors for use with ONNX Runtime.

Read the rest of the article here:
https://blogs.windows.com/msedgedev/2023/03/08/video-super-resolution-in-microsoft-edge/

More Info: https://www.microsoft.com/en-us/edge/features/enhance-video?form=MA13FJ