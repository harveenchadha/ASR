import hydra
from hydra.core.config_store import ConfigStore
from deepspeech_pytorch.enums import DecoderType
from deepspeech_pytorch.configs.inference_config import TranscribeConfig, LMConfig, ModelConfig
from deepspeech_pytorch.inference import transcribe
import pandas as pd
from tqdm import tqdm
cs = ConfigStore.instance()
cs.store(name="config", node=TranscribeConfig)


@hydra.main(config_name="config")
def hydra_main(cfg: TranscribeConfig):
    lm_configs = LMConfig(decoder_type=DecoderType.beam, lm_path = 'lm_v3.binary')  
    valid_path = pd.read_csv('/home/harveen.chadha/exp_5000/tarini_manifest.csv', header=None)
    model_path =  '/home/jupyter/exp_5000/train_continued/models/deepspeech_final.pth'
    files = valid_path.iloc[:,0].values
    #for file in tqdm(files):
    transcribe_cfg = TranscribeConfig(model=ModelConfig(cuda=True, model_path=model_path, use_half=True),
            lm=lm_configs, audio_path=files[0])

    transcribe(cfg=transcribe_cfg, save=True)


if __name__ == '__main__':
    hydra_main()
