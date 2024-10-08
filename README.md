# Finetune LLM

## Dataset Preparation

First, we have to get the data. Open Telegram, go to 'Setting' -> 'Advanced' -> 'Export Telegram Data' and unselect everything except 'Personal chats' and 'Private groups' (don't select 'Only my messages there'). As output format choose 'Machine-readable JSON'. It will result in `result.json`.

Use `prepare_dataset.py` to transform `result.json` to JSON with a list of sessions:

```bash
python prepare_dataset.py "./data/result.json" "./data/messages.json"
```

There are some flags available for this script, you can read more in `--help`:

```bash
python prepare_dataset.py --help
```

<details>
<summary>output</summary>

```
NAME
    prepare_dataset.py - Transforms chat histories from .json telegram export to .json with a list of sessions. Session is a list of messages, where each message is a dict with fields 'author' and 'text'.

SYNOPSIS
    prepare_dataset.py INPUT OUTPUT <flags>

DESCRIPTION
    Transforms chat histories from .json telegram export to .json with a list of sessions. Session is a list of messages, where each message is a dict with fields 'author' and 'text'.

POSITIONAL ARGUMENTS
    INPUT
        Type: str
        Path to .json telegram export, usually called result.json
    OUTPUT
        Type: str
        Path to output .json file

FLAGS
    -t, --target_name=TARGET_NAME
        Type: Optional[str | None]
        Default: None
        The name of the person to target. This person will be present in every session. If empty, will be tried to be detected from "Saved Messages"
    -l, --last_x_months=LAST_X_MONTHS
        Type: int
        Default: 24
        Number of last months to use messages from
    -s, --session_minutes_threshold=SESSION_MINUTES_THRESHOLD
        Type: int
        Default: 10
        Threshold in minutes where messages will belong to the same session
    -c, --concat_one_user_messages_delimeter=CONCAT_ONE_USER_MESSAGES_DELIMETER
        Type: str
        Default: '\n>>> '
        Users might type several messages one after each other. They are concatenated using this delimeter

NOTES
    You can also use flags syntax for POSITIONAL ARGUMENTS
```

</details>

If you are interested, Telegram have several types of messages which should be handled differently:

<details>
<summary>default text message</summary>

```
{
 "id": 123,
 "type": "message",
 "date": "2023-10-31T15:23:38",
 "date_unixtime": "1698746018",
 "from": "Username",
 "from_id": "user123",
 "text": "ты где?",
 "text_entities": [
  {
   "type": "plain",
   "text": "ты где?"
  }
 ]
}
```

</details>

<details>
<summary>multiple text entities</summary>

```
{
 "id": 345,
 "type": "message",
 "date": "2023-10-25T01:56:50",
 "date_unixtime": "1698179210",
 "from": "Username",
 "from_id": "user456",
 "text": [
  "California suspends GM Cruise's autonomous vehicle deployment | Hacker News\n",
  {
   "type": "link",
   "text": "https://news.ycombinator.com/item?id=38002752"
  }
 ],
 "text_entities": [
  {
   "type": "plain",
   "text": "California suspends GM Cruise's autonomous vehicle deployment | Hacker News\n"
  },
  {
   "type": "link",
   "text": "https://news.ycombinator.com/item?id=38002752"
  }
 ]
}
```

</details>

<details>
<summary>sticker</summary>

```
{
 "id": 789,
 "type": "message",
 "date": "2023-10-30T23:24:20",
 "date_unixtime": "1698688460",
 "from": "Username",
 "from_id": "user789",
 "file": "(File not included. Change data exporting settings to download.)",
 "thumbnail": "(File not included. Change data exporting settings to download.)",
 "media_type": "sticker",
 "sticker_emoji": "🤗",
 "width": 512,
 "height": 501,
 "text": "",
 "text_entities": []
}
```

</details>

## Training

Final version of models were trained with the parameters which are default in training scripts.

### LoRA fine-tune

To launch LoRA fine-tune with my default params, you will need GPU with 20GB VRAM. RTX 3090 is a good option for it's money. You may reduce `micro_batch_size` or `max_seq_length` if you want to lower the amount of VRAM required. To get full list of parameters, run:

```
python finetune_lora.py --help
```

To train LoRA, run:

```
python finetune_lora.py
```

### Full fine-tune

To list available params with their default values, run:

```
python finetune_full.py --help
```

To train:

```
torchrun --nnodes=1 --nproc_per_node=NUMBER_OF_GPUS finetune_full.py
```

To save model to HF, run:

```
python save_hf.py --model_name_or_path ./weights/full --hub_model_id jtz18/llama3-8b-jon --hf_token <YOUR_HF_TOKEN>
```

## Launching

Use [oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui). If you used LoRA, then clone [ehartford/dolphin-2.2.1-mistral-7b](https://huggingface.co/ehartford/dolphin-2.2.1-mistral-7b) or whatever model you are used as a base model and put trained LoRA connectors to `./loras/` folder within text-generation-webui. If you did full fine-tune, then copy training result to `./models/`.
