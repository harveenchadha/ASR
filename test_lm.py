import hydra
from hydra.core.config_store import ConfigStore
from deepspeech_pytorch.enums import DecoderType
from deepspeech_pytorch.configs.inference_config import EvalConfig, LMConfig, ModelConfig
from deepspeech_pytorch.testing import evaluate

cs = ConfigStore.instance()
cs.store(name="config", node=EvalConfig)


@hydra.main(config_name="config")
def hydra_main(cfg: EvalConfig):
    lm_configs = LMConfig(decoder_type=DecoderType.beam, lm_path = 'lm_v3.binary')

    model_path =  '/home/jupyter/exp_5000/train_continued/models/deepspeech_final.pth'
    eval_cfg = EvalConfig(
            model=ModelConfig(cuda=True, model_path=model_path, use_half=False),
            lm=lm_configs, test_manifest='tarini_manifest.csv')

    evaluate(cfg=eval_cfg)


if __name__ == '__main__':
    hydra_main()
