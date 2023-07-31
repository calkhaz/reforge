# Reforge

A real-time vulkan compute shader utility capable of chaining shaders together in a render graph

## Features
* Loading image formats supported via ffmpeg library
* Previewing compute shader(s) output in a real-time window
* Live updating when any shader in the graph is modified
* Parsing spirv for descriptor names

## Live-reloading Pipeline Graph Configuration on Modification
Reforge has its own simple and intuitive pipeline configuration

We can live reload this configuration file and have instant visual feedback

![graph-reload](https://github.com/calkhaz/reforge/assets/85903607/e17db279-a061-4ee1-91da-71e7f92b8cb9)

## Live-reloading Shaders on Modification
Similar to the config, we are able to reload any shader in the pipeline config by modifying it

Any syntax errors that happen when compiling your glsl will be outputted in detail

![shader-reload](https://github.com/calkhaz/reforge/assets/85903607/5076c628-edb4-4441-a0b9-502459e15834)

## Usage
```
Usage: reforge [OPTIONS] <input-file> [output-file]

Arguments:
  <input-file>   Required file to read from
  [output-file]  Optional jpg file to write to

Options:
      --width <WIDTH>
      --height <HEIGHT>
      --shader-format <SHADER_FORMAT>  Shader image format [default: rgba32f] [possible values: rgba8, rgba32f]
      --config <config>                Path to the pipeline configuration file
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
4. If you've vulkan SDK is not in your global path you'll need to source it before running:
```
source .../VulkanSDK/x.y.z/setup-env.sh
```
