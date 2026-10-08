import openai
import time
from tqdm import tqdm
import os
import re
import json
from modelscope.utils.constant import Tasks
from modelscope.pipelines import pipeline
from modelscope.preprocessors.multi_modal import OfaPreprocessor
from faithscore.llama_pre import load_llama, stage1_llama
from faithscore.utils import llava15, ofa
import nltk
from nltk.corpus import wordnet
# from nltk.stem import WordNetLemmatizer
NLTK_DATA = "/root/autodl-tmp/aims_benchmarks/nltk_3-8-1"
if NLTK_DATA not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DATA)
from nltk.stem import WordNetLemmatizer
_lemmatizer = WordNetLemmatizer()

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import torch 

path = os.path.dirname(__file__)
cur_path = os.path.dirname(path)
cur_path = os.path.join(cur_path, "faithscore")

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed


_OFA_MODEL = None


def _init_ofa_worker(model_dir):
    """
    Each worker process loads its own OFA model once.
    """
    global _OFA_MODEL

    preprocessor = OfaPreprocessor(model_dir=model_dir)

    _OFA_MODEL = pipeline(
        Tasks.visual_question_answering,
        model=model_dir,
        preprocessor=preprocessor,
    )


def _ofa_process_one_image(job):
    """
    Process one complete image and all of its atomic facts.

    job:
        {
            "idx": ...,
            "image": ...,
            "elements": [...]
        }
    """
    global _OFA_MODEL

    idx = job["idx"]
    image = job["image"]
    elements = job["elements"]

    sample_scores = []

    for element in elements:
        prompt = (
            "Statement: "
            + element
            + " Is this statement is right according to the image? "
              "Please answer yes or no."
        )

        output = ofa(
            False,
            _OFA_MODEL,
            prompt,
            image,
        )

        sample_scores.append(
            1 if "yes" in output.lower() else 0
        )

    return idx, image, sample_scores

class FaithScore():
    def __init__(self, vem_type, model_path=None, api_key=None, api_url=None,
                 llava_path=None, tokenzier_path=None, use_llama=False,
                 llama_path=None, resume_dir=None, max_api_retries=10, openai_num_workers=None, ofa_batch_size=None):
        openai.api_key = api_key
        openai.api_base = api_url
        self.use_llama = use_llama
        self.vem_path = model_path
        self.model_type = vem_type
        self.resume_dir = resume_dir
        self.max_api_retries = max_api_retries
        self.openai_num_workers = openai_num_workers # open num thres in parallel
        self.ofa_batch_size = ofa_batch_size

        model_list = ["ofa_ve", "ofa", "mplug", "blip2", "llava"]
        if vem_type not in model_list:
            raise ValueError(f"model type {vem_type} not in {model_list}")

        self.llava_path = llava_path
        if self.resume_dir:
            os.makedirs(self.resume_dir, exist_ok=True)

        if use_llama:
            if llama_path:
                self.llama, self.tokenizer = load_llama(llama_path)
            else:
                raise ValueError("please input the model path for llama")

    # -------------------------
    # checkpoint helpers
    # -------------------------
    def _cache_path(self, name):
        if not self.resume_dir:
            return None
        return os.path.join(self.resume_dir, name)

    def _load_jsonl_map(self, name):
        """Load JSONL records keyed by integer idx. Last record wins."""
        path = self._cache_path(name)
        cache = {}
        if not path or not os.path.exists(path):
            return cache
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    cache[int(rec["idx"])] = rec
                except Exception as e:
                    # A crash can leave only the final line half-written.
                    print(f"[resume] ignore malformed {name} line {line_no}: {e}")
        return cache

    def _append_jsonl(self, name, record):
        path = self._cache_path(name)
        if not path:
            return
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def call_openai(self, pts):
        for attempt in range(1, self.max_api_retries + 1):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": pts}],
                    temperature=0.2,
                )
                return response['choices'][0]['message']['content']
            except Exception as e:
                msg = str(e)
                # Retrying this will never succeed for the same prompt.
                if "context_length_exceeded" in msg:
                    raise
                if attempt == self.max_api_retries:
                    raise
                print(f"OpenAI error ({attempt}/{self.max_api_retries}): {e}")
                time.sleep(10)
    
    def stage1(self, answers):
        with open(os.path.join(cur_path, "prompts/prompt_label_des_ana.txt"), "r") as f:
            prompt_label_des_ana = f.read() + "\n\n"
        # auto_resume
        cache = self._load_jsonl_map("stage1.jsonl")
        des_ana = [None] * len(answers)
        reused = 0

        for idx in range(len(answers)):
            if idx in cache:
                des_ana[idx] = cache[idx]["output"]
                reused += 1

        if reused:
            print(f"[resume] Stage 1: reused {reused}/{len(answers)} cached samples")
        #
        def worker(idx):
            pts = (
                prompt_label_des_ana
                + answers[idx].replace("\n", " ")
                + "\nLabeled text: "
            )
            output = self.call_openai(pts).replace("\n", "")
            return idx, output

        pending = [
            idx for idx in range(len(answers))
            if des_ana[idx] is None
        ]

        with ThreadPoolExecutor(max_workers=self.openai_num_workers) as executor:
            futures = {
                executor.submit(worker, idx): idx
                for idx in pending
            }

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Stage 1"
            ):
                idx, output = future.result()

                des_ana[idx] = output

                self._append_jsonl("stage1.jsonl", {
                    "idx": idx,
                    "output": output,
                })
        # 
        return des_ana

    @staticmethod
    def _extract_descriptions(labeld_sub_sen):
        all_texts = []
        for ss in labeld_sub_sen:
            desc = ""
            pos_des = [m.start() for m in re.finditer(r"\[D\]", ss)]
            pos_ana = [m.start() for m in re.finditer(r"\[A\]", ss)]
            pos_seg = sorted(pos_des + pos_ana)
            for i in range(len(pos_seg)):
                if pos_seg[i] in pos_des:
                    if i == 0:
                        desc += ss[:pos_seg[i] - 1]
                    else:
                        desc += ss[pos_seg[i - 1] + 3:pos_seg[i] - 1]
            all_texts.append(desc.replace("\n", " "))
        return all_texts

    @staticmethod
    def _parse_atomic_response(facts):
        """Always return five aligned lists, even if the LLM omits a heading."""
        entity, relation, color, count, other = [], [], [], [], []
        for raw_line in facts.split("\n"):
            line = raw_line.strip()
            if line.startswith("Entities:"):
                content = line[len("Entities:"):].strip()
                entity = content.split(". ") if content else []
            elif line.startswith("Relations:"):
                content = line[len("Relations:"):].strip()
                relation = content.split(". ") if content else []
            elif line.startswith("Colors:"):
                content = line[len("Colors:"):].strip()
                color = content.split(". ") if content else []
            elif line.startswith("Counting:"):
                content = line[len("Counting:"):].strip()
                count = content.split(". ") if content else []
            elif line.startswith("Other attributes:"):
                content = line[len("Other attributes:"):].strip()
                other = content.split(". ") if content else []
        return entity, relation, color, count, other

    # def stage2(self, labeld_sub_sen):
    #     all_texts = self._extract_descriptions(labeld_sub_sen)

    #     with open(os.path.join(cur_path, "prompts/prompt_de_atomic.txt"), 'r') as f:
    #         prompt_de_atomic = f.read()

    #     nons = "Entities:\nRelations:\nColors:\nCounting:\nOther attributes:"
    #     # auto resume
    #     cache = self._load_jsonl_map("stage2.jsonl")
    #     results = [None] * len(all_texts)
    #     reused = 0

    #     for idx in range(len(all_texts)):
    #         if idx in cache:
    #             results[idx] = cache[idx]["response"]
    #             reused += 1

    #     if reused:
    #         print(f"[resume] Stage 2: reused {reused}/{len(all_texts)} cached samples")
    #     #
    #     #
    #     for idx, ans in enumerate(tqdm(all_texts, desc="Stage 2")):
    #         if results[idx] is not None:
    #             continue

    #         ans = ans.replace("\n", "")
    #         if ans == "":
    #             response = nons
    #         else:
    #             pts = prompt_de_atomic + "\nAnswer: " + ans
    #             response = self.call_openai(pts)
    #             if "Entities" not in response:
    #                 response = nons

    #         results[idx] = response
    #         self._append_jsonl("stage2.jsonl", {
    #             "idx": idx,
    #             "input": ans,
    #             "response": response,
    #         })

    #     Entities, Relations, Colors, Counting, Others = [], [], [], [], []
    #     for idx, facts in enumerate(results):
    #         entity, relation, color, count, other = self._parse_atomic_response(facts)
    #         Entities.append(entity)
    #         Relations.append(relation)
    #         Colors.append(color)
    #         Counting.append(count)
    #         Others.append(other)

    #     n = len(results)
    #     assert all(len(x) == n for x in [Entities, Relations, Colors, Counting, Others])

    #     hallucinations = [
    #         Entities[i] + Relations[i] + Colors[i] + Counting[i] + Others[i]
    #         for i in range(n)
    #     ]
    #     return hallucinations, Entities, Relations, Colors, Counting, Others
    def stage2(self, labeld_sub_sen):
        all_texts = self._extract_descriptions(labeld_sub_sen)

        with open(
            os.path.join(cur_path, "prompts/prompt_de_atomic.txt"), "r"
        ) as f:
            prompt_de_atomic = f.read()
        nons = "Entities:\nRelations:\nColors:\nCounting:\nOther attributes:"

        # =========================
        # Auto resume
        # =========================
        cache = self._load_jsonl_map("stage2.jsonl")
        results = [None] * len(all_texts)
        reused = 0

        for idx in range(len(all_texts)):
            if idx in cache:
                results[idx] = cache[idx]["response"]
                reused += 1

        if reused:
            print(
                f"[resume] Stage 2: reused "
                f"{reused}/{len(all_texts)} cached samples"
            )

        # =========================
        # Empty descriptions
        # No API call is necessary
        # =========================
        for idx, ans in enumerate(all_texts):
            if results[idx] is not None:
                continue

            ans = ans.replace("\n", "")

            if ans == "":
                results[idx] = nons

                self._append_jsonl(
                    "stage2.jsonl",
                    {
                        "idx": idx,
                        "input": ans,
                        "response": nons,
                    },
                )

        # =========================
        # OpenAI worker
        # =========================
        def worker(idx):
            ans = all_texts[idx].replace("\n", "")
            pts = prompt_de_atomic + "\nAnswer: " + ans

            response = self.call_openai(pts)

            return idx, ans, response

        # Only submit samples that:
        # 1. are not cached
        # 2. are not empty descriptions
        pending = [
            idx
            for idx in range(len(all_texts))
            if results[idx] is None
        ]

        # =========================
        # Concurrent OpenAI calls
        # =========================
        with ThreadPoolExecutor(max_workers=self.openai_num_workers) as executor:
            futures = {
                executor.submit(worker, idx): idx
                for idx in pending
            }

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Stage 2",
            ):
                idx, ans, response = future.result()

                # Keep alignment by original sample index
                results[idx] = response

                # Only the main thread writes checkpoint
                self._append_jsonl(
                    "stage2.jsonl",
                    {
                        "idx": idx,
                        "input": ans,
                        "response": response,
                    },
                )

        # =========================
        # Safety check
        # =========================
        assert all(x is not None for x in results), \
            "Stage 2 has unfinished samples"

        # =========================
        # Parse atomic facts
        # =========================
        Entities = []
        Relations = []
        Colors = []
        Counting = []
        Others = []

        for idx, facts in enumerate(results):
            entity, relation, color, count, other = \
                self._parse_atomic_response(facts)

            Entities.append(entity)
            Relations.append(relation)
            Colors.append(color)
            Counting.append(count)
            Others.append(other)

        # Every sample MUST have five aligned slots.
        # A missing category is represented by [].
        n = len(results)

        assert all(
            len(x) == n
            for x in [
                Entities,
                Relations,
                Colors,
                Counting,
                Others,
            ]
        ), (
            f"Stage 2 alignment error: "
            f"results={n}, "
            f"Entities={len(Entities)}, "
            f"Relations={len(Relations)}, "
            f"Colors={len(Colors)}, "
            f"Counting={len(Counting)}, "
            f"Others={len(Others)}"
        )

        hallucinations = [
            Entities[i]
            + Relations[i]
            + Colors[i]
            + Counting[i]
            + Others[i]
            for i in range(n)
        ]

        return (
            hallucinations,
            Entities,
            Relations,
            Colors,
            Counting,
            Others,
        )
    # def stage3(self, atomic_facts, images, img_path=None):
    #     if self.model_type == "ofa_ve":
    #         model = pipeline(
    #             Tasks.visual_entailment,
    #             model='iic/ofa_visual-entailment_snli-ve_large_en'
    #         )
    #     elif self.model_type == "ofa":
    #         model_dir = self.vem_path or "/blob/output/liyan/msr/models/iic/ofa_visual-question-answering_pretrain_large_en"
    #         preprocessor = OfaPreprocessor(model_dir=model_dir)
    #         model = pipeline(
    #             Tasks.visual_question_answering,
    #             model=model_dir,
    #             preprocessor=preprocessor
    #         )
    #     elif self.model_type == "llava":
    #         if not self.llava_path:
    #             raise ValueError("Please input path for LLaVA model.")
    #         from faithscore.llava15 import LLaVA
    #         model = LLaVA()
    #     else:
    #         raise ValueError(f"Unsupported model type: {self.model_type}")

    #     stage3_name = f"stage3_{self.model_type}.jsonl"
    #     cache = self._load_jsonl_map(stage3_name)
    #     fact_scores = [None] * len(atomic_facts)
    #     reused = 0
    #     # resume
    #     for idx in range(len(atomic_facts)):
    #         if idx in cache:
    #             cached_scores = cache[idx].get("fact_scores")
    #             # Only trust cache if it matches current number of atomic facts.
    #             if isinstance(cached_scores, list) and len(cached_scores) == len(atomic_facts[idx]):
    #                 fact_scores[idx] = cached_scores
    #                 reused += 1

    #     if reused:
    #         print(f"[resume] Stage 3: reused {reused}/{len(atomic_facts)} cached samples")

    #     for idx, elements in enumerate(tqdm(atomic_facts, desc="Stage 3")):
    #         if fact_scores[idx] is not None:
    #             continue

    #         image = os.path.join(img_path, images[idx]) if img_path else images[idx]
    #         sample_scores = []

    #         for element in elements:
    #             prompt = ('Statement: ' + element
    #                       + ' Is this statement is right according to the image? Please answer yes or no.')
    #             if self.model_type == "ofa_ve":
    #                 output = ofa(True, model, element, image)
    #             elif self.model_type == "ofa":
    #                 output = ofa(False, model, prompt, image)
    #             elif self.model_type == "llava":
    #                 output = llava15(image, prompt, model)
    #             sample_scores.append(1 if "yes" in output.lower() else 0)

    #         fact_scores[idx] = sample_scores
    #         self._append_jsonl(stage3_name, {
    #             "idx": idx,
    #             "image": image,
    #             "num_facts": len(elements),
    #             "fact_scores": sample_scores,
    #         })

    #     instance_score = [
    #         sum(scores) / len(scores) if len(scores) > 0 else 0
    #         for scores in fact_scores
    #     ]
    #     return sum(instance_score) / len(instance_score), fact_scores
    # def stage3(
    #     self,
    #     atomic_facts,
    #     images,
    #     img_path=None,
    #     num_workers=4,
    # ): # multiprocess
    #     # ==========================================
    #     # OFA multiprocessing
    #     # ==========================================
    #     if self.model_type == "ofa":
    #         model_dir = (self.vem_path
    #             or "/blob/output/liyan/msr/models/iic/"
    #             "ofa_visual-question-answering_pretrain_large_en"
    #         )

    #         stage3_name = "stage3_ofa.jsonl"

    #         # -------------------------------
    #         # Auto resume
    #         # -------------------------------
    #         cache = self._load_jsonl_map(stage3_name)

    #         fact_scores = [None] * len(atomic_facts)
    #         reused = 0

    #         for idx in range(len(atomic_facts)):
    #             if idx in cache:
    #                 cached_scores = cache[idx].get("fact_scores")

    #                 if (
    #                     isinstance(cached_scores, list)
    #                     and len(cached_scores)
    #                     == len(atomic_facts[idx])
    #                 ):
    #                     fact_scores[idx] = cached_scores
    #                     reused += 1

    #         if reused:
    #             print(
    #                 f"[resume] Stage 3: reused "
    #                 f"{reused}/{len(atomic_facts)} cached samples"
    #             )

    #         # -------------------------------
    #         # Construct image-level jobs
    #         # -------------------------------
    #         jobs = []

    #         for idx, elements in enumerate(atomic_facts):
    #             if fact_scores[idx] is not None:
    #                 continue

    #             image = (
    #                 os.path.join(img_path, images[idx])
    #                 if img_path
    #                 else images[idx]
    #             )

    #             # If no atomic facts, no OFA inference needed
    #             if len(elements) == 0:
    #                 fact_scores[idx] = []

    #                 self._append_jsonl(
    #                     stage3_name,
    #                     {
    #                         "idx": idx,
    #                         "image": image,
    #                         "num_facts": 0,
    #                         "fact_scores": [],
    #                     },
    #                 )

    #                 continue

    #             jobs.append(
    #                 {
    #                     "idx": idx,
    #                     "image": image,
    #                     "elements": elements,
    #                 }
    #             )

    #         print(
    #             f"[Stage 3] OFA multiprocessing: "
    #             f"{len(jobs)} images, "
    #             f"{num_workers} workers"
    #         )

    #         # ==========================================
    #         # IMPORTANT:
    #         # CUDA multiprocessing must use spawn
    #         # ==========================================
    #         ctx = mp.get_context("spawn")

    #         with ProcessPoolExecutor(
    #             max_workers=num_workers,
    #             mp_context=ctx,
    #             initializer=_init_ofa_worker,
    #             initargs=(model_dir,),
    #         ) as executor:

    #             futures = {
    #                 executor.submit(
    #                     _ofa_process_one_image,
    #                     job
    #                 ): job["idx"]
    #                 for job in jobs
    #             }

    #             for future in tqdm(
    #                 as_completed(futures),
    #                 total=len(futures),
    #                 desc=f"Stage 3 OFA x{num_workers}",
    #             ):
    #                 idx = futures[future]

    #                 try:
    #                     idx, image, sample_scores = future.result()

    #                 except Exception as e:
    #                     print(
    #                         f"[Stage 3 ERROR] idx={idx}: {repr(e)}"
    #                     )
    #                     raise

    #                 # Keep original order
    #                 fact_scores[idx] = sample_scores

    #                 # Main process alone writes checkpoint
    #                 self._append_jsonl(
    #                     stage3_name,
    #                     {
    #                         "idx": idx,
    #                         "image": image,
    #                         "num_facts": len(atomic_facts[idx]),
    #                         "fact_scores": sample_scores,
    #                     },
    #                 )

    #         # Ensure everything finished
    #         assert all(
    #             x is not None for x in fact_scores
    #         ), "Stage 3 has unfinished samples"

    #         instance_score = [
    #             sum(scores) / len(scores)
    #             if len(scores) > 0
    #             else 0
    #             for scores in fact_scores
    #         ]

    #         return (
    #             sum(instance_score) / len(instance_score),
    #             fact_scores,
    #         )

    #     # ==========================================
    #     # Original OFA-VE / LLaVA
    #     # ==========================================
    #     if self.model_type == "ofa_ve":
    #         model = pipeline(
    #             Tasks.visual_entailment,
    #             model="iic/ofa_visual-entailment_snli-ve_large_en"
    #         )

    #     elif self.model_type == "llava":
    #         if not self.llava_path:
    #             raise ValueError(
    #                 "Please input path for LLaVA model."
    #             )

    #         from faithscore.llava15 import LLaVA
    #         model = LLaVA()

    #     else:
    #         raise ValueError(
    #             f"Unsupported model type: {self.model_type}"
    #         )

    #     stage3_name = f"stage3_{self.model_type}.jsonl"
    #     cache = self._load_jsonl_map(stage3_name)

    #     fact_scores = [None] * len(atomic_facts)
    #     reused = 0

    #     for idx in range(len(atomic_facts)):
    #         if idx in cache:
    #             cached_scores = cache[idx].get("fact_scores")

    #             if (
    #                 isinstance(cached_scores, list)
    #                 and len(cached_scores)
    #                 == len(atomic_facts[idx])
    #             ):
    #                 fact_scores[idx] = cached_scores
    #                 reused += 1

    #     if reused:
    #         print(
    #             f"[resume] Stage 3: reused "
    #             f"{reused}/{len(atomic_facts)} cached samples"
    #         )

    #     for idx, elements in enumerate(
    #         tqdm(atomic_facts, desc="Stage 3")
    #     ):
    #         if fact_scores[idx] is not None:
    #             continue

    #         image = (
    #             os.path.join(img_path, images[idx])
    #             if img_path
    #             else images[idx]
    #         )

    #         sample_scores = []

    #         for element in elements:
    #             prompt = (
    #                 "Statement: "
    #                 + element
    #                 + " Is this statement is right according to the image? "
    #                 "Please answer yes or no."
    #             )

    #             if self.model_type == "ofa_ve":
    #                 output = ofa(
    #                     True,
    #                     model,
    #                     element,
    #                     image
    #                 )

    #             elif self.model_type == "llava":
    #                 output = llava15(
    #                     image,
    #                     prompt,
    #                     model
    #                 )

    #             sample_scores.append(
    #                 1 if "yes" in output.lower() else 0
    #             )

    #         fact_scores[idx] = sample_scores

    #         self._append_jsonl(
    #             stage3_name,
    #             {
    #                 "idx": idx,
    #                 "image": image,
    #                 "num_facts": len(elements),
    #                 "fact_scores": sample_scores,
    #             },
    #         )

    #     instance_score = [
    #         sum(scores) / len(scores)
    #         if len(scores) > 0
    #         else 0
    #         for scores in fact_scores
    #     ]

    #     return (
    #         sum(instance_score) / len(instance_score),
    #         fact_scores,
    #     )
    
    def stage3(self, atomic_facts, images, img_path=None, batch_size=4): # batched
        if self.model_type == "ofa_ve":
            model = pipeline(
                Tasks.visual_entailment,
                model='iic/ofa_visual-entailment_snli-ve_large_en'
            )

        elif self.model_type == "ofa":
            model_dir = self.vem_path

            preprocessor = OfaPreprocessor(model_dir=model_dir)

            model = pipeline(
                Tasks.visual_question_answering,
                model=model_dir,
                preprocessor=preprocessor
            )

        elif self.model_type == "llava":
            if not self.llava_path:
                raise ValueError("Please input path for LLaVA model.")

            from faithscore.llava15 import LLaVA
            model = LLaVA()

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        import types
        import torch
        import torch.nn.functional as F

        pad_token_id = preprocessor.tokenizer.pad_token_id
        print(f"[OFA] pad_token_id = {pad_token_id}")

        def _ofa_batch_with_padding(self_pipeline, data_list):
            batch_data = {
                "nsentences": len(data_list),
                "net_input": {},
                "decoder_prompts": [],
                "samples": [],
            }

            # Batch net_input
            for key in data_list[0]["net_input"].keys():
                values = [item["net_input"][key] for item in data_list]

                if key == "input_ids":
                    max_len = max(v.shape[1] for v in values)

                    padded_values = []

                    for v in values:
                        pad_len = max_len - v.shape[1]

                        if pad_len > 0:
                            v = F.pad(
                                v,
                                (0, pad_len),
                                value=pad_token_id,
                            )

                        padded_values.append(v)

                    batch_data["net_input"][key] = torch.cat(
                        padded_values,
                        dim=0,
                    )

                else:
                    batch_data["net_input"][key] = torch.cat(
                        values,
                        dim=0,
                    )

            # Merge decoder_prompts
            for item in data_list:
                if "decoder_prompts" in item:
                    batch_data["decoder_prompts"].extend(
                        list(item["decoder_prompts"])
                    )

            # Merge samples
            for item in data_list:
                if "samples" in item:
                    batch_data["samples"].extend(
                        item["samples"]
                    )

            return batch_data


        model._batch = types.MethodType(
            _ofa_batch_with_padding,
            model,
        )
        stage3_name = f"stage3_{self.model_type}_batch.jsonl"
        cache = self._load_jsonl_map(stage3_name)

        fact_scores = [None] * len(atomic_facts)
        reused = 0

        # ==========================
        # Resume
        # ==========================
        for idx in range(len(atomic_facts)):
            if idx in cache:
                cached_scores = cache[idx].get("fact_scores")

                if (
                    isinstance(cached_scores, list)
                    and len(cached_scores) == len(atomic_facts[idx])
                ):
                    fact_scores[idx] = cached_scores
                    reused += 1

        if reused:
            print(
                f"[resume] Stage 3: reused "
                f"{reused}/{len(atomic_facts)} cached samples"
            )

        # ==========================
        # OFA: flatten all facts
        # ==========================
        if self.model_type == "ofa":

            jobs = []

            for idx, elements in enumerate(atomic_facts):
                if fact_scores[idx] is not None:
                    continue

                image = (
                    os.path.join(img_path, images[idx])
                    if img_path
                    else images[idx]
                )

                # reserve slots
                fact_scores[idx] = [None] * len(elements)

                for fact_idx, element in enumerate(elements):
                    prompt = (
                        "Statement: "
                        + element
                        + " Is this statement is right according to the image? "
                        "Please answer yes or no."
                    )

                    jobs.append({
                        "idx": idx,
                        "fact_idx": fact_idx,
                        "image": image,
                        "prompt": prompt,
                    })
            # ==========================
            # Batch inference
            # ==========================
            checked_batch = False

            for start in tqdm(
                range(0, len(jobs), batch_size),
                desc=f"Stage 3 OFA batch={batch_size}"
            ):
                batch_jobs = jobs[start:start + batch_size]

                batch_inputs = [
                    {
                        "image": job["image"],
                        "text": job["prompt"],
                    }
                    for job in batch_jobs
                ]

                # ==========================
                # CHECK: inspect preprocess tensor shapes once
                # ==========================
                if not checked_batch:
                    print("\n[CHECK] Inspecting OFA preprocessed inputs for the first batch...")

                    preprocessed_list = []

                    for i, inp in enumerate(batch_inputs):
                        x = model.preprocess(inp)
                        preprocessed_list.append(x)

                        print(f"\n[CHECK] sample={i}, idx={batch_jobs[i]['idx']}, fact_idx={batch_jobs[i]['fact_idx']}")

                        for key, value in x.items():
                            if isinstance(value, dict):
                                for k, v in value.items():
                                    if torch.is_tensor(v):
                                        print(f"  {key}.{k}: shape={tuple(v.shape)}, dtype={v.dtype}")
                                    else:
                                        print(f"  {key}.{k}: type={type(v)}")
                            elif torch.is_tensor(value):
                                print(f"  {key}: shape={tuple(value.shape)}, dtype={value.dtype}")
                            else:
                                print(f"  {key}: type={type(value)}")

                    print("\n[CHECK] Tensor shapes across this batch:")

                    if "net_input" in preprocessed_list[0]:
                        for key in preprocessed_list[0]["net_input"].keys():
                            values = [x["net_input"].get(key) for x in preprocessed_list]

                            if all(torch.is_tensor(v) for v in values):
                                shapes = [tuple(v.shape) for v in values]
                                print(f"  net_input.{key}: {shapes}")

                    # ==========================
                    # CHECK: group by src_tokens length
                    # ==========================
                    if (
                        "net_input" in preprocessed_list[0]
                        and "src_tokens" in preprocessed_list[0]["net_input"]
                    ):
                        from collections import defaultdict

                        length_groups = defaultdict(list)

                        for i, x in enumerate(preprocessed_list):
                            length = x["net_input"]["src_tokens"].shape[-1]
                            length_groups[length].append(i)

                        print("\n[CHECK] src_tokens length groups:")

                        for length, indices in sorted(length_groups.items()):
                            print(f"  length={length}: samples={indices}")

                        # Try one same-length mini-batch if possible
                        candidate = None

                        for length, indices in length_groups.items():
                            if len(indices) >= 2:
                                candidate = indices[:min(len(indices), 4)]
                                break

                        if candidate is not None:
                            same_len_inputs = [batch_inputs[i] for i in candidate]

                            print(
                                f"\n[CHECK] Trying same-length batch with "
                                f"{len(candidate)} samples, "
                                f"src_tokens length="
                                f"{preprocessed_list[candidate[0]]['net_input']['src_tokens'].shape[-1]}"
                            )

                            try:
                                same_len_outputs = model(
                                    same_len_inputs,
                                    batch_size=len(same_len_inputs)
                                )

                                print(
                                    f"[CHECK] Same-length batch SUCCESS, "
                                    f"num_outputs={len(same_len_outputs)}"
                                )

                            except Exception as e:
                                print(
                                    f"[CHECK] Same-length batch FAILED: "
                                    f"{repr(e)}"
                                )
                        else:
                            print(
                                "\n[CHECK] No two samples in the first batch "
                                "have the same src_tokens length, so same-length "
                                "batch test was skipped."
                            )

                    checked_batch = True

                    print("\n[CHECK] Finished OFA batch diagnostics.\n")
                # ==========================
                # Original batch inference
                # ==========================
                outputs = model(
                    batch_inputs,
                    batch_size=batch_size
                )
                print("[DEBUG OUTPUTS]", outputs)
                print("[DEBUG OUTPUT TYPE]", type(outputs))

                for i, output in enumerate(outputs[:4]):
                    print(f"[DEBUG output {i}] type={type(output)} value={output}")
                    if isinstance(output, dict):
                        print(f"[DEBUG text {i}] type={type(output.get('text'))} value={output.get('text')}")

                for job, output in zip(batch_jobs, outputs):
                    idx = job["idx"]
                    fact_idx = job["fact_idx"]

                    output_text = output["text"][0]

                    fact_scores[idx][fact_idx] = (
                        1 if output_text.lower() == "yes" else 0
                    )

            # ==========================
            # Save completed images
            # ==========================
            for idx, scores in enumerate(fact_scores):

                if scores is None:
                    continue

                # Empty atomic fact sample is valid
                if len(scores) == 0:
                    complete = True
                else:
                    complete = all(x is not None for x in scores)

                if not complete:
                    continue

                image = (
                    os.path.join(img_path, images[idx])
                    if img_path
                    else images[idx]
                )

                # Don't duplicate already resumed entries
                if idx not in cache:
                    self._append_jsonl(
                        stage3_name,
                        {
                            "idx": idx,
                            "image": image,
                            "num_facts": len(atomic_facts[idx]),
                            "fact_scores": scores,
                        }
                    )

        # ==========================
        # Other verifier types:
        # keep original sequential path
        # ==========================
        else:
            for idx, elements in enumerate(
                tqdm(atomic_facts, desc="Stage 3")
            ):
                if fact_scores[idx] is not None:
                    continue

                image = (
                    os.path.join(img_path, images[idx])
                    if img_path
                    else images[idx]
                )

                sample_scores = []

                for element in elements:
                    prompt = (
                        "Statement: "
                        + element
                        + " Is this statement is right according to the image? "
                        "Please answer yes or no."
                    )

                    if self.model_type == "ofa_ve":
                        output = ofa(True, model, element, image)

                    elif self.model_type == "llava":
                        output = llava15(image, prompt, model)

                    sample_scores.append(
                        1 if "yes" in output.lower() else 0
                    )

                fact_scores[idx] = sample_scores

                self._append_jsonl(
                    stage3_name,
                    {
                        "idx": idx,
                        "image": image,
                        "num_facts": len(elements),
                        "fact_scores": sample_scores,
                    }
                )

        instance_score = [
            sum(scores) / len(scores)
            if len(scores) > 0
            else 0
            for scores in fact_scores
        ]

        return (
            sum(instance_score) / len(instance_score),
            fact_scores
        )
    #

    def faithscore(self, answers, images):
        labeld_sub_sen = self.stage1(answers)
        atomic_facts, Entities, Relations, Colors, Counting, Others = self.stage2(labeld_sub_sen)
        score, fact_scores = self.stage3(atomic_facts, images, batch_size=self.ofa_batch_size)
        sentence_score = self.sentence_faithscore(
            Entities, Relations, Colors, Counting, Others,
            self.labeled_sub(labeld_sub_sen), fact_scores
        )
        return score, sentence_score

    def sentence_faithscore(self, Entities, Relations, Colors, Counting, Others, all_texts, fact_scores):
        Entities_recog = []

        for sample_idx, ents in enumerate(Entities):
            entities = []

            for ent_idx, ent in enumerate(ents):
                ent4sen = []

                if not isinstance(ent, str):
                    print(
                        "\n[ERROR] Non-string entity:",
                        f"sample_idx={sample_idx}",
                        f"ent_idx={ent_idx}",
                        f"type={type(ent)}",
                        f"value={repr(ent)}",
                        f"all_ents={repr(ents)}",
                    )
                    raise TypeError(
                        f"Entity must be str, got {type(ent)}: {repr(ent)}"
                    )

                sentence = nltk.sent_tokenize(ent)
                tags = nltk.pos_tag(nltk.word_tokenize(sentence[0]))

                for tag in tags:
                    if tag[1] in [
                        'NN', 'NNS', 'JJ', 'NNP',
                        'VBG', 'JJR', 'NNPS', 'RB', 'DT'
                    ]:
                        ent4sen.append(tag[0])

                if len(ent4sen) < 1:
                    fallback_tokens = [
                        tok for tok, pos in tags
                        if any(c.isalnum() for c in tok)
                    ]

                    if len(fallback_tokens) < 1:
                        raise RuntimeError(
                            f"Cannot recognize entity tokens: {ent}; tags={tags}"
                        )

                    ent4sen.append(fallback_tokens[-1])

                    print(
                        f"[Entity POS fallback] "
                        f"{ent!r} -> {ent4sen[-1]!r}; tags={tags}"
                    )

                entities.append(ent4sen[-1])

            if len(entities) != len(ents):
                raise RuntimeError(
                    f"Entity recognition length mismatch: "
                    f"sample={sample_idx}, "
                    f"{len(entities)} vs {len(ents)}"
                )

            Entities_recog.append(entities)

        if len(Entities_recog) != len(Entities):
            raise RuntimeError(
                f"Entities_recog length mismatch: "
                f"{len(Entities_recog)} vs {len(Entities)}"
            )

        entity_scores, relation_scores, color_scores, count_scores, other_scores = [], [], [], [], []
        for i in range(len(fact_scores)):
            entity_scores.append(fact_scores[i][:len(Entities[i])])
            relation_scores.append(fact_scores[i][len(Entities[i]): len(Entities[i]) + len(Relations[i])])
            color_scores.append(fact_scores[i][len(Entities[i]) + len(Relations[i]): len(Entities[i]) + len(Relations[i]) + len(Colors[i])])
            count_scores.append(fact_scores[i][len(Entities[i]) + len(Relations[i]) + len(Colors[i]): len(Entities[i]) + len(Relations[i]) + len(Colors[i]) + len(Counting[i])])
            other_scores.append(fact_scores[i][len(Entities[i]) + len(Relations[i]) + len(Colors[i]) + len(Counting[i]):])

        sentence_scores = []
        for id1, ins in enumerate(all_texts):
            sentence_score = []
            for id2, sub_sen in enumerate(all_texts[id1]):
                flag = True
                for id3, ee in enumerate(Entities_recog[id1]):
                    if ee in sub_sen and entity_scores[id1][id3] != 1:
                        flag = False
                    for id4, rel in enumerate(relation_scores[id1]):
                        if ee in sub_sen and ee in Relations[id1][id4] and rel != 1:
                            flag = False
                    for id4, rel in enumerate(color_scores[id1]):
                        if ee in sub_sen and ee in Colors[id1][id4] and rel != 1:
                            flag = False
                    for id4, rel in enumerate(count_scores[id1]):
                        if ee in sub_sen and ee in Counting[id1][id4] and rel != 1:
                            flag = False
                    for id4, rel in enumerate(other_scores[id1]):
                        if ee in sub_sen and ee in Others[id1][id4] and rel != 1:
                            flag = False
                sentence_score.append(flag)
            sentence_scores.append(sentence_score)

        score4sen = [sum(ss)/len(ss) if len(ss) > 0 else 1 for ss in sentence_scores]
        return sum(score4sen)/len(score4sen)

    def labeled_sub(self, des_ana):
        all_texts = []
        for ss in des_ana:
            desc = []
            pos_des = [m.start() for m in re.finditer(r"\[D\]", ss)]
            pos_ana = [m.start() for m in re.finditer(r"\[A\]", ss)]
            pos_seg = sorted(pos_des + pos_ana)
            for i in range(len(pos_seg)):
                if pos_seg[i] in pos_des:
                    if i == 0:
                        desc.append(ss[:pos_seg[i] - 1])
                    else:
                        desc.append(ss[pos_seg[i - 1] + 3:pos_seg[i] - 1])
            all_texts.append(desc)
        return all_texts
