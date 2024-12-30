import json
import os
import argparse
import openai
import copy
import logging
import time
from tqdm import *
from data_processor import EcspellProcessor

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# chatGPT API
class ChatGPT4CSC(object):
    def __init__(self,key_file,message_file=None, prompt_examples = None, key_id=0):
        self.key_file=key_file
        self.key_id = key_id
        self.openai_key=None
        self.messages={"role": "system", "content": "你是一个中文拼写错误修改助手"}
        self.message_file=message_file##directory to store the chat messages if neccesary
        self.prefix = "请修改句子中的拼写错误，要求修改后的句子和原句长度相同, 如果句子中没有错误，请直接照抄原句。\n"
        self.icl_prompt = ""
        if prompt_examples is not None:
            for example in prompt_examples:
                src = "".join(example.src)
                trg = "".join(example.trg)
                self.icl_prompt += src + '\n' + trg + '\n'
    
    def get_api_key(self):
        with open(self.key_file, 'r', encoding='utf-8') as f:
            self.openai_key = json.load(f)[self.key_id]["api_key"]
            openai.api_key=self.openai_key
    
    def gptCorrect(self,src):
        result=openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                        {"role": "system", "content": "你是一个中文拼写错误修改助手"},
                        {"role": "user", "content":  self.prefix + self.icl_prompt + src},
                    ]
                )
        return result.get("choices")[0].get("message").get("content") ## the response of chatgpt

class Metrics:
    @staticmethod
    def compute(src_sents, trg_sents, prd_sents):
        def difference(src, trg):
            ret = copy.deepcopy(src)
            for i, (src_char, trg_char) in enumerate(zip(src, trg)):
                if src_char!= trg_char:
                    ret[i] = "(" + src_char + "->" + trg_char + ")"

            return "".join(ret)
        def equals(src,trg):
            if len(src)!=len(trg):
                return False
            for i,(st,tt) in enumerate(zip(src,trg)):
                # we do not consider the punctuation
                if st not in ['.','。',',','，','?','？',':','：'] and st!=tt:
                    return False
            return True
        pos_sents, neg_sents, tp_sents, fp_sents, fn_sents, prd_pos_sents, prd_neg_sents, wp_sents  = [], [], [], [], [], [], [], []
        ## wp_sents are the positive examples corrected in a wrong way (s!=t&p!=t&p!=s)
        for s, t, p in zip(src_sents, trg_sents, prd_sents):
            # For positive examples
            if not equals(s,t):
                pos_sents.append(difference(s, t))
                if equals(t,p):
                    tp_sents.append(difference(s, t))
                if equals(s,p):
                    fn_sents.append(difference(s, t))
                if (not equals(p,t)) and (not equals(p,s)):
                    wp_sents.append(difference(s,t))
            # For negative examples
            else:
                neg_sents.append(difference(s, t))
                if not equals(p,t):
                    fp_sents.append(difference(t, p))
            # For predictions
            if not equals(p,s):
                prd_pos_sents.append(difference(s, p))
            if equals(p,s):
                prd_neg_sents.append(difference(s, p))

        p = 1.0 * len(tp_sents) / len(prd_pos_sents)
        r = 1.0 * len(tp_sents) / len(pos_sents)
        f1 = 2.0 * (p * r) / (p + r + 1e-12)
        fpr = 1.0 * (len(fp_sents) + 1e-12) / (len(neg_sents) + 1e-12)

        return p, r, f1, fpr, tp_sents, fp_sents, fn_sents, wp_sents

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_chatgpt",action="store_true")
    parser.add_argument("--load_messages",action="store_true")
    parser.add_argument("--key_file",type=str,default="../envs/openai_key",help="the file containing the api key")
    parser.add_argument("--message_file",type=str,default="model/messages_odw.json",help="the file to store the chat messages")
    parser.add_argument("--data_dir", type=str, default="../data/",
                        help="Directory to contain the input data for all tasks.")
    parser.add_argument("--task_name", type=str, default="ecspell",
                        help="Name of the training task.")
    parser.add_argument("--train_on", type=str, default="law",help="Choose a set where the few shot learning examples come from")
    parser.add_argument("--test_on", type=str, default="law",help="Choose a dev set.")
    parser.add_argument("--output_dir", type=str, default="../model/",
                        help="Directory to output predictions and checkpoints.")

    parser.add_argument("--few_shot", type=int, default =0, help='if we apply few shot in context learning and the number of shots.')
    parser.add_argument("--key_id", default=0,type=int)
    
    args = parser.parse_args()
    if args.use_chatgpt:
        processors = {
            "ecspell": EcspellProcessor,
        }
        task_name = args.task_name.lower()
        if task_name not in processors:
            raise ValueError("Task not found: %s" % task_name)
        processor = processors[task_name]()
        # get examples for icl
        if args.few_shot != 0:
            examples = processor.get_train_examples(os.path.join(args.data_dir,args.task_name),division=args.train_on)
            prompt_examples = examples[:args.few_shot]
        else:
            prompt_examples = None
        # initialize the chatgpt
        chat=ChatGPT4CSC(key_file=args.key_file, prompt_examples=prompt_examples, key_id=args.key_id)
        chat.get_api_key()
        logger.info("api_key: %s",chat.openai_key)
        # load the data
        test_examples=processor.get_test_examples(os.path.join(args.data_dir,args.task_name),division=args.test_on)##[example(.src,.trg,.guid)]
        all_preds=[]
        all_srcs=[]
        all_trgs=[]
        messages=[]
        for i,example in enumerate(tqdm(test_examples,desc="Test")):
            while(1):
                logger.info("-----{}-----".format(i))
                try:
                    src=example.src #[t1,t2,...,tn]
                    trg=example.trg
                    prediction=chat.gptCorrect("".join(src))
                except:
                    logger.info("---\n")
                    time.sleep(5)
                    continue
                if i%100 == 0 or (i%10==0 and i<100):
                    logger.info("src: %s \n prediction: %s", "".join(src),prediction)
                all_preds.append(list(prediction))
                all_srcs.append(src)
                all_trgs.append(trg)
                messages.append({"src":"".join(src),"trg":"".join(trg), "pred":prediction})
                break
        with open(args.message_file, 'w', encoding='utf-8') as f:
            json.dump(messages,f, ensure_ascii=False,indent=4)
    else:
        all_preds=[]
        all_srcs=[]
        all_trgs=[]
        if args.load_messages:
            with open(args.message_file,'r',encoding='utf-8') as f:
                messages=json.load(f)
        for i, message in enumerate(messages):
            all_preds.append(list(message['pred']))
            all_srcs.append(list(message['src']))
            all_trgs.append(list(message['trg']))

    p, r, f1, fpr, tp, fp, fn, wp = Metrics.compute(all_srcs, all_trgs, all_preds)
    output_tp_file = os.path.join(args.output_dir, "sents.tp")
    with open(output_tp_file, "w", encoding="utf-8") as writer:
        for line in tp:
            writer.write(line + "\n")
    output_fp_file = os.path.join(args.output_dir, "sents.fp")
    with open(output_fp_file, "w",encoding='utf-8') as writer:
        for line in fp:
            writer.write(line + "\n")
    output_fn_file = os.path.join(args.output_dir, "sents.fn")
    with open(output_fn_file, "w",encoding="utf-8") as writer:
        for line in fn:
            writer.write(line + "\n")
    output_wp_file = os.path.join(args.output_dir, "sents.wp")
    with open(output_wp_file, "w",encoding='utf-8') as writer:
        for line in wp:
            writer.write(line + "\n")

    result = {
        "eval_p": p * 100,
        "eval_r": r * 100,
        "eval_f1": f1 * 100,
        "eval_fpr": fpr * 100,
    }
    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    with open(output_eval_file, "a") as writer:
        logger.info("***** Eval results *****")
        writer.write(
            "eval precision = %.2f | eval recall = %.2f | eval f1 = %.2f | eval fp rate = %.2f\n"
            % (result["eval_p"],
            result["eval_r"],
            result["eval_f1"],
            result["eval_fpr"]))
        for key in sorted(result.keys()):
            logger.info("%s = %s", key, str(result[key]))
if __name__ == "__main__":
    main()