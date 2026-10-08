from faithscore.framework import FaithScore
import argparse
import os
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--answer_path', type=str, required=True)
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--openai_key', type=str, default="api.key")
    parser.add_argument('--openai_url', type=str, default="api.key")
    parser.add_argument('--vem_type', type=str, choices=["ofa", "ofa-ve", "llava"], default="llava")
    parser.add_argument('--model_path', type=str, default=None,
                        help='Local VEM model directory, recommended for OFA.')
    parser.add_argument('--llava_path', type=str, default=".cache/factscore/")
    parser.add_argument('--llama_path', type=str, default=".cache/factscore/")
    parser.add_argument('--use_llama', action='store_true')
    parser.add_argument('--debug_number', type=int, default=-1)
    parser.add_argument('--resume_dir', type=str, default=None,
                        help='Checkpoint directory. Default: <answer_path>.faithscore_resume')
    parser.add_argument('--max_api_retries', type=int, default=10)
    parser.add_argument('--output_path', type=str, default=None)
    
    parser.add_argument('--openai_num_workers', type=int, default=10)
    parser.add_argument('--ofa_batch_size', type=int, default=1)
    args = parser.parse_args()

    args.resume_dir = os.path.dirname(args.answer_path)
    images, answers = [], []
    with open(args.answer_path, encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if args.debug_number >= 0 and idx == args.debug_number:
                break
            line = json.loads(line)
            answers.append(line["model_answer"])
            images.append(os.path.join(args.image_dir, line["image"]))

    resume_dir = args.resume_dir or (args.answer_path + ".faithscore_resume")
    os.makedirs(resume_dir, exist_ok=True)
    print(f"[resume] checkpoint dir: {resume_dir}")
    # init
    score = FaithScore(
        vem_type=args.vem_type,
        model_path=args.model_path,
        api_key=args.openai_key,
        api_url=args.openai_url,
        llava_path=args.llava_path,
        use_llama=args.use_llama,
        llama_path=args.llama_path,
        resume_dir=resume_dir,
        max_api_retries=args.max_api_retries,
        openai_num_workers=args.openai_num_workers,
        ofa_batch_size=args.ofa_batch_size,
    )
    # import ipdb;ipdb.set_trace()
    f, sentence_f = score.faithscore(answers, images)
    print(f"Faithscore is {f}. Sentence-level faithscore is {sentence_f}.")

    output_path = args.output_path or os.path.join(resume_dir, "final_score.json")
    with open(output_path, "w", encoding="utf-8") as out:
        json.dump({
            "answer_path": args.answer_path,
            "num_samples": len(answers),
            "vem_type": args.vem_type,
            "faithscore": f,
            "sentence_faithscore": sentence_f,
        }, out, indent=2, ensure_ascii=False)
    print(f"Saved final result to: {output_path}")
