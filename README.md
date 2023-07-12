# Reforge

A real-time vulkan compute shader utility capable of chaining shaders together in a render graph

## Features
* Loading image formats suported via ffmpeg library
* Previewing compute shader(s) output in a real-time window
* Live updating when any shader in the graph is modified
* Parsing spirv for descriptor names

## Usage
```
Usage: reforge [OPTIONS] <input-file>

Arguments:
  <input-file>

Options:
      --width <WIDTH>
      --height <HEIGHT>
      --shader-format <SHADER_FORMAT>  Shader image format [default: rgba32f] [possible values: rgba8, rgba32f]
      --num-frames <NUM_FRAMES>        Number of frame-in-flight to be used when displaying to the swapchain [default: 2]
  -h, --help                           Print help
```

## Setup
1. Download the VulkanSDK here: https://vulkan.lunarg.com/sdk/home
2. Install rust - On Mac, I recommend getting the latest rust by doing:
```
brew install rustup
rustup-init
```
3. cargo build
4. If you've vulkan SDK is not in your global path you'll need to source it before runing:
```
source .../VulkanSDK/x.y.z/setup-env.sh
```
