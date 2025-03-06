python ckpt/download_ckpt.py \
    --source hf \
    --repo_id microsoft/wavlm-large \
    --filename pytorch_model.bin \
    --save_path ./WavLM-Large.pt

python ckpt/download_ckpt.py \
    --source hf \
    --repo_id facebook/w2v-bert-2.0 \
    --filename model.safetensors \
    --save_path \
    ./codec_ckpt/hub/models--facebook--w2v-bert-2.0/model.safetensors

python ckpt/download_ckpt.py \
     --source hf \
     --repo_id HKUSTAudio/xcodec2 \
     --filename ckpt/epoch=4-step=1400000.ckpt \
     --save_path ./codec_ckpt/epoch=4-step=1400000.ckpt