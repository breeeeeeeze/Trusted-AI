# TRUSTED AI
## A Discord chat generation bot based on character-level RNN

Trusted-AI is discord bot that will imitate any chat it has been trained on. It can also log any messages sent in its channels so they can later be used as training data. Currently this is in the "janky fun project" stage so expect bugs and inefficiencies.

### Requirements

- tensorflow>=2.8
- CUDA and cuDNN highly recommended
- discord>=1.7.3
- regex
- python_dotenv
- pandas
- tqdm

### How to run

Duplicate `.env_example` and `config_example.json` into **`.env`** and **`config.json`**. If you want to run the bot, insert your bot token into the .env file and the server ID and channel IDs the bot should be active in aswell as the owner ID (for control commands, currently only for the shutdown command).

For the bot, launch `bot.py`. For training use `train.py`.

### Message logger

The bot will automatically start logging all messages, including message content, id, channel id, author id and attachment (only the first if multiple) to the export file defined in the config file. {date} can be used to automatically split the export files by date. The files generated can directly be used as input for training or further processed (e.g. split by author and/or channel).

### Training process

#### Input data

The output of the message logger can be used as training data. If historical data is desired, this can be gathered using a chat export tool **(WARNING: Automating user accounts violates Discord ToS)** or using the data request feature in the privacy settings.

Either way the data processor expects csv files in a folder called `data` with at least a *Contents* row. If the `removeRowIfAttachment` setting is set to true, an *Attachments row is also necessary.

#### Preprocessing

The data is automatically preprocessed according to the options in the config file. Note: If you don't filter out pings, the bot will randomly ping people.

#### Training

Training is automatic, using the parameters specified in the config. Several models are possible, when using a dynamic model specify the number of hidden layers in the config. The training will output the best and last checkpoints of the model and the vocabulary so it can be used by the bot. Tensorboard files and a pickled history object is also outputted.

This process will take quite a bit of time, even on a high-end GPU.

### Chat prediction

Add any trained models in the prediction section of the config file. If all settings are set correctly it will automatically load the stored models and is ready to predict. Use `ai.<model_name> <seed_text>` or `ai.predict <model_name> <seed_text>` to generate a prediction. Add `-t <decimal_number>` between name and seed to alter the behaviour of the model, numbers below 1 will generate more predictable text but may always give the same prediction for a given seed, numbers above 1 give more chaotic results that become less believable or even complete garble.

### Current limitations

- There may be bugs hidden everywhere
- There is currently no way to disable the logger or predictor bot parts separately
- The input data has to be located in a folder called `data`
- There is no way to specify any settings other than model type for the prediction, so everything in the learn section of the config has to stay the same between training and prediction for all models. If not, model building will fail.
- Vanilla emoji filtering is broken
- Depending on the seed it can start predicting in the middle of words


