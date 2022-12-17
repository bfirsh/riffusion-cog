# riffusion

[riffusion](https://www.riffusion.com/about) as a Cog model.

First, [install Cog](https://github.com/replicate/cog) and download the weights:

    cog run script/download-weights

Then, you can run predictions:

    cog predict -i prompt="funky synth solo"
    
And push it to Replicate:

    cog push r8.im/hmartiro/riffusion
