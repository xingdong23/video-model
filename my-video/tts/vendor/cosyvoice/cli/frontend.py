# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from functools import partial
from typing import Generator
import json
import jieba.posseg as pseg
import onnxruntime
import torch
import numpy as np
import whisper
from typing import Callable
import torchaudio.compliance.kaldi as kaldi
import torchaudio
import os
import re
import inflect
import pickle
import logging

logger = logging.getLogger(__name__)


class CachedZhNormalizer:
    def __init__(self, remove_erhua=False, full_to_half=False, overwrite_cache=False):
        self.remove_erhua = remove_erhua
        self.full_to_half = full_to_half
        self.cache_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "cache"
        )
        self.cache_file = os.path.join(
            self.cache_dir,
            f"zh_normalizer_cache_erhua{remove_erhua}_full{full_to_half}.pkl",
        )
        self._normalizer = None
        self._load_or_create_normalizer(overwrite_cache)

    def _load_or_create_normalizer(self, overwrite_cache):
        try:
            if not overwrite_cache and os.path.exists(self.cache_file):
                logger.info(f"正在从缓存加载中文规范化器: {self.cache_file}")
                with open(self.cache_file, "rb") as f:
                    self._normalizer = pickle.load(f)
                logger.info("中文规范化器加载完成")
            else:
                logger.info("正在创建新的中文规范化器...")
                os.makedirs(self.cache_dir, exist_ok=True)
                self._normalizer = ZhNormalizer(
                    remove_erhua=self.remove_erhua,
                    full_to_half=self.full_to_half,
                    overwrite_cache=True,
                )
                logger.info(f"正在保存中文规范化器到缓存: {self.cache_file}")
                with open(self.cache_file, "wb") as f:
                    pickle.dump(self._normalizer, f)
                logger.info("中文规范化器缓存保存完成")
        except Exception as e:
            logger.error(f"加载/创建中文规范化器时出错: {str(e)}")
            logger.info("回退到直接创建中文规范化器...")
            self._normalizer = ZhNormalizer(
                remove_erhua=self.remove_erhua,
                full_to_half=self.full_to_half,
                overwrite_cache=True,
            )

    def __getattr__(self, name):
        return getattr(self._normalizer, name)


try:
    import ttsfrd

    use_ttsfrd = True
except ImportError:
    print("failed to import ttsfrd, use WeTextProcessing instead")
    try:
        from tn.chinese.normalizer import Normalizer as ZhNormalizer
        from tn.english.normalizer import Normalizer as EnNormalizer
    except ImportError:
        print("failed to import WeTextProcessing, use identity normalizer instead")

        class ZhNormalizer:
            def __init__(self, *args, **kwargs):
                pass

            def normalize(self, text):
                return text

        class EnNormalizer:
            def __init__(self, *args, **kwargs):
                pass

            def normalize(self, text):
                return text

    use_ttsfrd = False
from cosyvoice.utils.file_utils import logging
from cosyvoice.utils.frontend_utils import (
    contains_chinese,
    replace_blank,
    replace_corner_mark,
    remove_bracket,
    spell_out_number,
    split_paragraph,
    is_only_punctuation,
)

splits = {
    "，",
    "。",
    "？",
    "！",
    ",",
    ".",
    "?",
    "!",
    "~",
    ":",
    "：",
    "—",
    "…",
}


def get_first(text):
    pattern = "[" + "".join(re.escape(sep) for sep in splits) + "]"
    text = re.split(pattern, text)[0].strip()
    return text


def split(todo_text):
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if todo_text[-1] not in splits:
        todo_text += "。"
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)
    todo_texts = []
    while 1:
        if i_split_head >= len_text:
            break  # 结尾一定有标点，所以直接跳出即可，最后一段在上次已加入
        if todo_text[i_split_head] in splits:
            i_split_head += 1
            todo_texts.append(todo_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
    return todo_texts


def cut5(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    if len(inps) < 2:
        return inp
    opts = []
    summ = 0
    tmp_str = ""
    for i in range(len(inps)):
        summ += len(inps[i])
        tmp_str += inps[i]
        if summ > 10:
            summ = 0
            opts.append(tmp_str)
            tmp_str = ""
    if tmp_str != "":
        opts.append(tmp_str)
    # print(opts)
    if len(opts) > 1 and len(opts[-1]) < 50:  ##如果最后一个太短了，和前一个合一起
        opts[-2] = opts[-2] + opts[-1]
        opts = opts[:-1]
    return opts


def text_normalize(text):
    """
    对文本进行归一化处理
    :param text:
    :return:
    """
    from .zh_normalization import TextNormalizer

    # ref: https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/paddlespeech/t2s/frontend/zh_normalization
    tx = TextNormalizer()
    sentences = tx.normalize(text)
    # print(sentences)

    _txt = "".join(sentences)
    # 替换掉除中文之外的所有字符
    # _txt = re.sub(
    #     r"[^\u4e00-\u9fa5，。！？、]+", "", _txt
    # )

    return _txt


def remove_chinese_punctuation(text):
    """
    移除文本中的中文标点符号 [：；！（），【】『』「」《》－‘“’”:,;!\(\)\[\]><\-] 替换为 ，
    :param text:
    :return:
    """
    chinese_punctuation_pattern = (
        r"[：；！（），【】『』「」《》－‘“’”:,;!\(\)\[\]><\-·]"
    )
    text = re.sub(chinese_punctuation_pattern, "，", text)
    # 使用正则表达式将多个连续的句号替换为一个句号
    text = re.sub(r"[。，]{2,}", "。", text)
    # 删除开头和结尾的 ， 号
    text = re.sub(r"^，|，$", "", text)
    return text


def normalize_zh(text):
    return process_ddd(text_normalize(remove_chinese_punctuation(text)))


def process_ddd(text):
    """
    处理“地”、“得” 字的使用，都替换为“的”
    依据：地、得的使用，主要是在动词和形容词前后，本方法没有严格按照语法替换，因为时常遇到用错的情况。
    另外受 jieba 分词准确率的影响，部分情况下可能会出漏掉。例如：小红帽疑惑地问
    :param text: 输入的文本
    :return: 处理后的文本
    """
    word_list = [(word, flag) for word, flag in pseg.cut(text, use_paddle=False)]
    # print(word_list)
    processed_words = []
    for i, (word, flag) in enumerate(word_list):
        if word in ["地", "得"]:
            # Check previous and next word's flag
            # prev_flag = word_list[i - 1][1] if i > 0 else None
            # next_flag = word_list[i + 1][1] if i + 1 < len(word_list) else None

            # if prev_flag in ['v', 'a'] or next_flag in ['v', 'a']:
            if flag in ["uv", "ud"]:
                processed_words.append("的")
            else:
                processed_words.append(word)
        else:
            processed_words.append(word)

    return "".join(processed_words)


class CosyVoiceFrontEnd:

    def __init__(
        self,
        get_tokenizer: Callable,
        feat_extractor: Callable,
        campplus_model: str,
        speech_tokenizer_model: str,
        spk2info: str = "",
        allowed_special: str = "all",
    ):
        self.tokenizer = get_tokenizer()
        self.feat_extractor = feat_extractor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        option.intra_op_num_threads = 1
        self.campplus_session = onnxruntime.InferenceSession(
            campplus_model, sess_options=option, providers=["CPUExecutionProvider"]
        )
        self.speech_tokenizer_session = onnxruntime.InferenceSession(
            speech_tokenizer_model,
            sess_options=option,
            providers=[
                (
                    "CUDAExecutionProvider"
                    if torch.cuda.is_available()
                    else "CPUExecutionProvider"
                )
            ],
        )
        if os.path.exists(spk2info):
            self.spk2info = torch.load(spk2info, map_location=self.device)
        else:
            self.spk2info = {}
        self.allowed_special = allowed_special
        self.use_ttsfrd = use_ttsfrd
        if self.use_ttsfrd:
            self.frd = ttsfrd.TtsFrontendEngine()
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            assert (
                self.frd.initialize(
                    "{}/../../pretrained_models/CosyVoice-ttsfrd/resource".format(
                        ROOT_DIR
                    )
                )
                is True
            ), "failed to initialize ttsfrd resource"
            self.frd.set_lang_type("pinyinvg")
        else:
            self.zh_tn_model = CachedZhNormalizer(
                remove_erhua=False,
                full_to_half=False,
                overwrite_cache=False,  # 设置为True可以强制重建缓存
            )
            self.en_tn_model = EnNormalizer()
            self.inflect_parser = inflect.engine()

    def _extract_text_token(self, text):
        if isinstance(text, Generator):
            logging.info(
                "get tts_text generator, will return _extract_text_token_generator!"
            )
            # NOTE add a dummy text_token_len for compatibility
            return self._extract_text_token_generator(text), torch.tensor(
                [0], dtype=torch.int32
            ).to(self.device)
        else:
            text_token = self.tokenizer.encode(
                text, allowed_special=self.allowed_special
            )
            text_token = torch.tensor([text_token], dtype=torch.int32).to(self.device)
            text_token_len = torch.tensor([text_token.shape[1]], dtype=torch.int32).to(
                self.device
            )
            return text_token, text_token_len

    def _extract_text_token_generator(self, text_generator):
        for text in text_generator:
            text_token, _ = self._extract_text_token(text)
            for i in range(text_token.shape[1]):
                yield text_token[:, i : i + 1]

    def _extract_speech_token(self, speech):
        assert (
            speech.shape[1] / 16000 <= 30
        ), "do not support extract speech token for audio longer than 30s"
        feat = whisper.log_mel_spectrogram(speech, n_mels=128)
        speech_token = (
            self.speech_tokenizer_session.run(
                None,
                {
                    self.speech_tokenizer_session.get_inputs()[0]
                    .name: feat.detach()
                    .cpu()
                    .numpy(),
                    self.speech_tokenizer_session.get_inputs()[1].name: np.array(
                        [feat.shape[2]], dtype=np.int32
                    ),
                },
            )[0]
            .flatten()
            .tolist()
        )
        speech_token = torch.tensor([speech_token], dtype=torch.int32).to(self.device)
        speech_token_len = torch.tensor([speech_token.shape[1]], dtype=torch.int32).to(
            self.device
        )
        return speech_token, speech_token_len

    def _extract_spk_embedding(self, speech):
        feat = kaldi.fbank(speech, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding = (
            self.campplus_session.run(
                None,
                {
                    self.campplus_session.get_inputs()[0]
                    .name: feat.unsqueeze(dim=0)
                    .cpu()
                    .numpy()
                },
            )[0]
            .flatten()
            .tolist()
        )
        embedding = torch.tensor([embedding]).to(self.device)
        return embedding

    def _extract_speech_feat(self, speech):
        speech_feat = (
            self.feat_extractor(speech).squeeze(dim=0).transpose(0, 1).to(self.device)
        )
        speech_feat = speech_feat.unsqueeze(dim=0)
        speech_feat_len = torch.tensor([speech_feat.shape[1]], dtype=torch.int32).to(
            self.device
        )
        return speech_feat, speech_feat_len

    def text_normalize(self, text, split=True, text_frontend=True):
        if isinstance(text, Generator):
            logging.info("get tts_text generator, will skip text_normalize!")
            return [text]
        if text_frontend is False:
            return [text] if split is True else text
        text = text.strip()
        if self.use_ttsfrd:
            texts = [
                i["text"]
                for i in json.loads(self.frd.do_voicegen_frd(text))["sentences"]
            ]
            text = "".join(texts)
        else:
            if contains_chinese(text):
                text = self.zh_tn_model.normalize(text)
                text = text.replace("\n", "。")
                text = replace_blank(text)
                text = replace_corner_mark(text)
                text = text.replace(".", "。")
                text = text.replace(" - ", "，")
                text = remove_bracket(text)
                text = re.sub(r"[，,、]+$", "。", text)
                texts = list(
                    split_paragraph(
                        text,
                        partial(
                            self.tokenizer.encode, allowed_special=self.allowed_special
                        ),
                        "zh",
                        token_max_n=80,
                        token_min_n=60,
                        merge_len=20,
                        comma_split=False,
                    )
                )
            else:
                text = self.en_tn_model.normalize(text)
                text = spell_out_number(text, self.inflect_parser)
                texts = list(
                    split_paragraph(
                        text,
                        partial(
                            self.tokenizer.encode, allowed_special=self.allowed_special
                        ),
                        "en",
                        token_max_n=80,
                        token_min_n=60,
                        merge_len=20,
                        comma_split=False,
                    )
                )
        texts = [i for i in texts if not is_only_punctuation(i)]
        return texts if split is True else text

    def text_normalize_stream(self, text, split=True):
        text = text.strip()
        if contains_chinese(text):
            # text = self.frd.get_frd_extra_info(text, 'input').replace("\n", "")
            text += ".。"
            text = text.replace("\n", "")
            text = normalize_zh(text)
            text = replace_blank(text)
            text = replace_corner_mark(text)
            text = text.replace(".", "、")
            text = text.replace(" - ", "，")
            text = remove_bracket(text)
            texts = [
                i
                for i in split_paragraph(
                    text,
                    partial(
                        self.tokenizer.encode, allowed_special=self.allowed_special
                    ),
                    "zh",
                    token_max_n=30,
                    token_min_n=20,
                    merge_len=15,
                    comma_split=True,
                )
            ]
        else:
            text += "."
            text = spell_out_number(text, self.inflect_parser)
            texts = [
                i
                for i in split_paragraph(
                    text,
                    partial(
                        self.tokenizer.encode, allowed_special=self.allowed_special
                    ),
                    "en",
                    token_max_n=30,
                    token_min_n=20,
                    merge_len=15,
                    comma_split=True,
                )
            ]
        if split is False:
            return text
        return texts

    def text_normalize_instruct(self, text, split=True):
        text = text.strip()
        if contains_chinese(text):
            # text = self.frd.get_frd_extra_info(text, 'input').replace("\n", "")
            text += ".。"
            text = text.replace("\n", "")
            # text = normalize_zh(text)
            text = replace_blank(text)
            text = replace_corner_mark(text)
            text = text.replace(".", "、")
            text = text.replace(" - ", "，")
            text = remove_bracket(text)
            texts = [
                i
                for i in split_paragraph(
                    text,
                    partial(
                        self.tokenizer.encode, allowed_special=self.allowed_special
                    ),
                    "zh",
                    token_max_n=80,
                    token_min_n=60,
                    merge_len=20,
                    comma_split=False,
                )
            ]
        else:
            text += "."
            text = spell_out_number(text, self.inflect_parser)
            texts = [
                i
                for i in split_paragraph(
                    text,
                    partial(
                        self.tokenizer.encode, allowed_special=self.allowed_special
                    ),
                    "en",
                    token_max_n=80,
                    token_min_n=60,
                    merge_len=20,
                    comma_split=False,
                )
            ]
        if split is False:
            return text
        return texts

    def frontend_sft(self, tts_text, spk_id):
        tts_text_token, tts_text_token_len = self._extract_text_token(tts_text)
        embedding = self.spk2info[spk_id]["embedding"]
        model_input = {
            "text": tts_text_token,
            "text_len": tts_text_token_len,
            "llm_embedding": embedding,
            "flow_embedding": embedding,
        }
        return model_input

    def frontend_zero_shot(
        self, tts_text, prompt_text, prompt_speech_16k, resample_rate
    ):
        tts_text_token, tts_text_token_len = self._extract_text_token(tts_text)
        prompt_text_token, prompt_text_token_len = self._extract_text_token(prompt_text)
        prompt_speech_resample = torchaudio.transforms.Resample(
            orig_freq=16000, new_freq=resample_rate
        )(prompt_speech_16k)
        speech_feat, speech_feat_len = self._extract_speech_feat(prompt_speech_resample)
        speech_token, speech_token_len = self._extract_speech_token(prompt_speech_16k)
        if resample_rate == 24000:
            # cosyvoice2, force speech_feat % speech_token = 2
            token_len = min(int(speech_feat.shape[1] / 2), speech_token.shape[1])
            speech_feat, speech_feat_len[:] = (
                speech_feat[:, : 2 * token_len],
                2 * token_len,
            )
            speech_token, speech_token_len[:] = speech_token[:, :token_len], token_len
        embedding = self._extract_spk_embedding(prompt_speech_16k)
        model_input = {
            "text": tts_text_token,
            "text_len": tts_text_token_len,
            "prompt_text": prompt_text_token,
            "prompt_text_len": prompt_text_token_len,
            "llm_prompt_speech_token": speech_token,
            "llm_prompt_speech_token_len": speech_token_len,
            "flow_prompt_speech_token": speech_token,
            "flow_prompt_speech_token_len": speech_token_len,
            "prompt_speech_feat": speech_feat,
            "prompt_speech_feat_len": speech_feat_len,
            "llm_embedding": embedding,
            "flow_embedding": embedding,
        }
        return model_input

    def frontend_cross_lingual(self, tts_text, prompt_speech_16k, resample_rate):
        model_input = self.frontend_zero_shot(
            tts_text, "", prompt_speech_16k, resample_rate
        )
        # in cross lingual mode, we remove prompt in llm
        del model_input["prompt_text"]
        del model_input["prompt_text_len"]
        del model_input["llm_prompt_speech_token"]
        del model_input["llm_prompt_speech_token_len"]
        return model_input

    def frontend_instruct(self, tts_text, spk_id, instruct_text):
        model_input = self.frontend_sft(tts_text, spk_id)
        # in instruct mode, we remove spk_embedding in llm due to information leakage
        del model_input["llm_embedding"]
        instruct_text_token, instruct_text_token_len = self._extract_text_token(
            instruct_text + "<endofprompt>"
        )
        model_input["prompt_text"] = instruct_text_token
        model_input["prompt_text_len"] = instruct_text_token_len
        return model_input

    def frontend_instruct2(
        self, tts_text, instruct_text, prompt_speech_16k, resample_rate
    ):
        model_input = self.frontend_zero_shot(
            tts_text,
            instruct_text + "<|endofprompt|>",
            prompt_speech_16k,
            resample_rate,
        )
        del model_input["llm_prompt_speech_token"]
        del model_input["llm_prompt_speech_token_len"]
        return model_input

    def frontend_vc(self, source_speech_16k, prompt_speech_16k, resample_rate):
        prompt_speech_token, prompt_speech_token_len = self._extract_speech_token(
            prompt_speech_16k
        )
        prompt_speech_resample = torchaudio.transforms.Resample(
            orig_freq=16000, new_freq=resample_rate
        )(prompt_speech_16k)
        prompt_speech_feat, prompt_speech_feat_len = self._extract_speech_feat(
            prompt_speech_resample
        )
        embedding = self._extract_spk_embedding(prompt_speech_16k)
        source_speech_token, source_speech_token_len = self._extract_speech_token(
            source_speech_16k
        )
        model_input = {
            "source_speech_token": source_speech_token,
            "source_speech_token_len": source_speech_token_len,
            "flow_prompt_speech_token": prompt_speech_token,
            "flow_prompt_speech_token_len": prompt_speech_token_len,
            "prompt_speech_feat": prompt_speech_feat,
            "prompt_speech_feat_len": prompt_speech_feat_len,
            "flow_embedding": embedding,
        }
        return model_input
