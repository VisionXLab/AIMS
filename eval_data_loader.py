import json
import os
import random

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

def load_json_or_jsonl(filepath):
    if filepath.endswith(".json"):
        all_data = json.load(open(filepath, "r", encoding="utf-8"))
    elif filepath.endswith(".jsonl"):
        all_data = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                all_data.append(json.loads(line.strip()))
    return all_data

class COCODataSet(Dataset):
    def __init__(self, data_path, trans, debug_number=-1):
        self.data_path = data_path
        self.trans = trans

        img_files = os.listdir(self.data_path)
        random.shuffle(img_files)
        self.img_files = img_files
        if debug_number != -1:
            self.img_files = self.img_files[:debug_number]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_file = self.img_files[index]
        img_id = int(img_file.split(".jpg")[0][-6:])

        image = Image.open(os.path.join(self.data_path, img_file)).convert("RGB")
        image = self.trans(image)  
        image_path = os.path.join(self.data_path, img_file) # qwenvl use image_path

        return {"img_id": img_id, "image": image, "image_path": image_path}

class MMHALDataSet(Dataset): # Abandon
    def __init__(self, data_path, json_file, trans, debug_number=-1):
        self.data_path = data_path
        self.trans = trans
        json_data = json.load(open(json_file, 'r'))
        self.all_data = []

        for item in json_data:
            item["image_src"] = os.path.join(self.data_path, os.path.basename(item["image_src"]))
            self.all_data.append(item)
            # import ipdb; ipdb.set_trace()
        random.shuffle(self.all_data)
        if debug_number != -1:
            self.all_data = self.all_data[:debug_number]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        data_item = self.all_data[index]
        question_type = question_type
        question_topic = question_topic
        image_id = image_id
        image_src = image_src 
        image_content = image_content 
        question = question 
        gt_answer = gt_answer
        model_answer = model_answer

        # image = Image.open(os.path.join(self.data_path, img_file)).convert("RGB")
        # image = self.trans(image)  
        image_path = os.path.join(self.data_path, img_file) # qwenvl use image_path

        return {"img_id": img_id, "image_path": image_src, "question": question}

class POPEChatDataSet(Dataset):
    def __init__(self, pope_path, data_path, trans):
        self.pope_path = pope_path
        self.data_path = data_path
        self.trans = trans

        image_list, query_list, label_list = [], [], []

        for q in open(pope_path, "r"):
            line = json.loads(q)
            image_list.append(line["image"])
            query_list.append(line["text"])
            label_list.append(line["label"])

        for i in range(len(label_list)):
            for j in range(len(label_list[i])):
                if label_list[i][j] == "no":
                    label_list[i][j] = 0
                else:
                    label_list[i][j] = 1

        assert len(image_list) == len(query_list)
        assert len(image_list) == len(label_list)

        self.image_list = image_list
        self.query_list = query_list
        self.label_list = label_list

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        image_path = os.path.join(self.data_path, self.image_list[index])
        raw_image = Image.open(image_path).convert("RGB")
        if self.trans is not None:
            image = self.trans(raw_image)
        else:
            image = transforms.ToTensor()(raw_image)
        query = self.query_list[index]
        label = self.label_list[index]

        return {"image": image, "query": query, "label": label, "image_path": image_path}

class MMEDataSet(Dataset):
    def __init__(self, data_path, trans, debug_number=-1):
        self.data_path = data_path
        self.trans = trans
        self.items = []

        dims = os.listdir(data_path)
        for dim in os.listdir(data_path):

            dim_dir = os.path.join(data_path, dim)
            if os.path.isdir(os.path.join(dim_dir, "images")):
                # artwork / color / existence ...
                img_dir = os.path.join(dim_dir, "images")
                qa_dir = os.path.join(dim_dir, "questions_answers_YN")

                for file in os.listdir(img_dir):
                    if file.endswith(".txt"):
                        continue
                    elif not (
                        file.endswith(".jpg")
                        or file.endswith(".png")
                        or file.endswith(".jpeg")
                    ):
                        import ipdb; ipdb.set_trace()
                    stem = os.path.splitext(file)[0]
                    self.items.append({
                        "dimension": dim,
                        "image_path": os.path.join(img_dir, file),
                        "qa_path": os.path.join(qa_dir, stem + ".txt"),
                        "img_name": file,
                    })
            else:
                # commonsense_reasoning / numerical_calculation ...
                for file in os.listdir(dim_dir):
                    if file.endswith(".txt"):
                        continue
                    elif not (
                        file.endswith(".jpg")
                        or file.endswith(".png")
                        or file.endswith(".jpeg")
                    ):
                        import ipdb; ipdb.set_trace()

                    stem = os.path.splitext(file)[0]

                    self.items.append({
                        "dimension": dim,
                        "image_path": os.path.join(dim_dir, file),
                        "qa_path": os.path.join(dim_dir, stem + ".txt"),
                        "img_name": file,
                    })

        random.shuffle(self.items)
        if debug_number != -1:
            self.items = self.items[:debug_number]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]

        image = Image.open(item["image_path"]).convert("RGB")
        image_tensor = self.trans(image)
        # img_id = str(item['img_name'].split(".")[0])


        questions = []
        answers = []

        with open(item["qa_path"], "r") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                q, a = line.rsplit("\t", 1)

                questions.append(q.strip())
                answers.append(a.strip())

        return {
            "image": image_tensor,
            # "img_id": img_id,
            "image_path": item["image_path"],
            "img_name": item["img_name"],
            "dimension": item["dimension"],
            "questions": questions,   # 长度一般为2
            "answers": answers,       # 对应GT
        }