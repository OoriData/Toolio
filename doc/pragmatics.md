Practical notes

# Managing the models cache

# Offline operation

Many users of Toolio will be referencing HuggingFace paths to models, especially in the [HF MLX community](). By defsult whenever you reference a model in this way it is downloaded and cached on your drive, so you needn't download it again in future, unless it changes upstream (theer's a new checkpoint).

This also means you should be able to use Toolio with already downloaded models when you are not connected to the Internet, but some details about how models are loaded add a wrinkle to matters. If you reference any model by its HF path, even if it has already been cached, there will be internet access, and things will hang or fail if you are offline.

## Environment variables

Try the environment variable `HF_DATASETS_OFFLINE="1"`. Also try setting `HF_HOME` to a cache folder with predownloaded the necessary models & datasets.

## Specifying the cache local directory

One solution ot this is to explicitly load the local cache directory rather than the HF path. An easy first step is to scan the cache to see what's there:

```sh
mlx_lm.manage --scan --pattern ""
```

This scan will reveal the HF path as well as local cache dirs for all cached models. Here is one example line from my case:

```
mlx-community/Hermes-2-Theta-Llama-3-8B-4bit model             4.5G        6 6 weeks ago   6 weeks ago   /Users/username/.cache/huggingface/hub/models--mlx-community--Hermes-2-Theta-Llama-3-8B-4bit 
```

Focusing on the first and last columns, this says that `mlx-community/Hermes-2-Theta-Llama-3-8B-4bit` is cached at `/Users/username/.cache/huggingface/hub/models--mlx-community--Hermes-2-Theta-Llama-3-8B-4bit`. If you check that directory, you'll find a `snapshots`, and one or more hash-named directories under that. You can specify one of these instead of the HF path, e.g.

```sh
toolio_server --model="/Users/username/.cache/huggingface/hub/models--mlx-community--Hermes-2-Theta-Llama-3-8B-4bit/snapshots/a1b2c3d4e5f6g7h8etc/"
```

Loading in this way will avoid any attempts to connect to the internet.

## Saving a pretrained model

If you'd ratehr have a location of your choice, avoid the hash-based names, etc., you can use…

## Further exploration

`local_files_only` argument to `snapshot_download` (used by mlx_lm under the hood)—_(bool, optional, defaults to False) — If True, avoid downloading the file and return the path to the local cached file if it exists._

