
#!/usr/bin/env python

# -*- coding: utf-8 -*-

import rospy

from sensor_msgs.msg import Image as ROSImage

from std_msgs.msg import String

from cv_bridge import CvBridge, CvBridgeError

import cv2

from PIL import Image

import sys

import torch

from transformers import AutoModel, AutoTokenizer

import threading

import numpy as np



rospy.loginfo("Using Python executable: %s", sys.executable)



OBJECT_MAP = {

    "烟雾弹": "Smokegrenade", "smoke": "Smokegrenade",

    "手雷": "Grenade", "grenade": "Grenade",

    "军用手电筒": "Torch", "手电筒": "Torch", "手电": "Torch",

    "弹夹": "Magazine", "弹匣": "Magazine", "杂志": "Magazine",

}

KEYWORDS = {

    "Torch": ["torch"],

    "Smokegrenade": ["smoke"],

    "Grenade": ["grenade"],

    "Magazine": ["mag", "magazine"],

}





def parse_yolo_labels(text):

    out = {}

    for part in (text or "").split(";"):

        if "|" not in part:

            continue

        seg = part.split("|")

        if len(seg) != 5 or not seg[0]:

            continue

        try:

            u = float(seg[1].split("=")[1])

            v = float(seg[2].split("=")[1])

            d = float(seg[3].split("=")[1])

            a = float(seg[4].split("=")[1])

        except Exception:

            continue

        out[seg[0]] = (u, v, d, a)

    return out





class ImageProcessorNode:

    def __init__(self, default_bbox_prompt):

        self.model = None

        self.tokenizer = None



        self.latest_cv_image = None

        self.frame_lock = threading.Lock()

        self.latest_labels = {}
        self.labels_ts = {}

        self.labels_lock = threading.Lock()



        self.bbox_prompt = default_bbox_prompt

        self.prompt_lock = threading.Lock()

        self.new_bbox_request = False

        self.target_en = None
        self.side_req = ""



        self.bridge = CvBridge()

        self.model_output_pub = rospy.Publisher("model_output", String, queue_size=10)



        model_file = '/root/inference/FM9G4B-V'

        try:

            rospy.loginfo("loading model from %s ...", model_file)

            self.model = AutoModel.from_pretrained(

                model_file,

                trust_remote_code=True,

                attn_implementation='sdpa',

                torch_dtype=torch.bfloat16

            )

            self.model = self.model.eval().to(device='cuda', dtype=torch.bfloat16)

            self.tokenizer = AutoTokenizer.from_pretrained(model_file, trust_remote_code=True)

            rospy.loginfo("模型和tokenizer加载成功。")

        except Exception as e:

            rospy.logerr("模型加载失败: %s", e)

            self.model = None

            self.tokenizer = None



        if self.model is not None and self.tokenizer is not None:

            self.image_sub = rospy.Subscriber(

                "/yolo_annotated_image", ROSImage, self.image_callback, queue_size=1)

            self.labels_sub = rospy.Subscriber(

                "/yolo_labels", String, self.labels_callback, queue_size=1)

            rospy.loginfo("成功订阅彩色图像和标签话题。")



    def labels_callback(self, msg):
        now = rospy.Time.now().to_sec()
        parsed = parse_yolo_labels(msg.data)
        with self.labels_lock:
            for k, v in parsed.items():
                self.latest_labels[k] = v
                self.labels_ts[k] = now
    def image_callback(self, data):

        try:

            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

        except CvBridgeError as e:

            rospy.logerr("彩色图像转换出错: %s", e)

            return

        with self.frame_lock:

            self.latest_cv_image = cv_image



    def build_prompt(self, text):

        t = text.strip().lower()

        for key, en in OBJECT_MAP.items():

            if key in t:

                hint = ""

                if en == "Smokegrenade":

                    hint = "（SMOKE 是军绿色圆柱体，别和 Grenade 搞混）"

                elif en == "Grenade":

                    hint = "（Grenade 是椭圆形带引信，别和 SMOKE 搞混）"

                prompt = ("图中每个绿框上方有文字标签，格式为 类别名(横坐标,纵坐标,深度m,角度)。"

                          "请找到以 %s 开头的标签%s，原样完整复制这整条标签，"

                          "并用左侧或右侧说明方位。只输出标签和方位，不要输出其他内容。"

                          % (en, hint))

                return prompt, en

        req = ""
        if ("右侧" in t) or ("右边" in t) or ("往右" in t) or ("放到右" in t):
            req = "右侧"
        elif ("左侧" in t) or ("左边" in t) or ("往左" in t) or ("放到左" in t):
            req = "左侧"
        self.side_req = req
        return text.strip(), None



    def process_latest_frame(self):

        if not self.new_bbox_request:

            return



        with self.frame_lock:

            if self.latest_cv_image is None:

                rospy.logwarn("当前没有有效彩色图像帧，跳过处理。")

                return

            cv_image = self.latest_cv_image.copy()



        try:

            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

            pil_image = Image.fromarray(rgb_image)

        except Exception as e:

            rospy.logerr("彩色图像转换错误: %s", e)

            return



        with self.prompt_lock:

            current_bbox_prompt = self.bbox_prompt

        msgs = [{'role': 'user', 'content': [pil_image, current_bbox_prompt]}]

        try:

            model_res = str(self.model.chat(image=None, msgs=msgs, tokenizer=self.tokenizer))

            rospy.loginfo("模型原始回答:\n%s", model_res)

        except Exception as e:

            rospy.logerr("调用大模型时出错: %s", e)

            return



        low = model_res.lower()

        final = model_res

        target_en = self.target_en



        if target_en:
            lab = None
            with self.labels_lock:
                _ts = self.labels_ts.get(target_en, 0.0)
                if rospy.Time.now().to_sec() - _ts < 5.0:
                    lab = self.latest_labels.get(target_en)
            if lab is not None:
                side = ""
                if getattr(self, "side_req", ""):
                    side = " " + self.side_req
                elif "左" in model_res:
                    side = " 左侧"
                elif "右" in model_res:
                    side = " 右侧"
                if not side:
                    side = " 左侧" if int(lab[0]) < 640 else " 右侧"
                u, v, d, a = lab
                final = "%s(%d,%d,%.2fm,%.2f)%s" % (target_en, int(u), int(v), d, a, side)
                rospy.loginfo("YOLO-TRUTH-REFORM: %s", final)
            else:
                final = "NO-TARGET-DETECTED"
                rospy.logwarn("target %s not in view, motion blocked", target_en)
        out = String()

        out.data = final

        self.model_output_pub.publish(out)

        rospy.loginfo("已发布到 model_output: %s", final)



    def command_input_thread(self):

        while not rospy.is_shutdown():

            try:

                new_prompt = input("\n请输入指令（例如：抓取烟雾弹）：")

                if new_prompt.strip():

                    prompt, en = self.build_prompt(new_prompt)

                    with self.prompt_lock:

                        self.bbox_prompt = prompt

                    self.target_en = en

                    rospy.loginfo("已识别目标: %s", en or "未匹配，按原话发送")

                    self.new_bbox_request = True

                    self.process_latest_frame()

                else:

                    rospy.loginfo("输入为空。")

            except Exception as e:

                rospy.logerr("终端输入错误: %s", e)

            rospy.sleep(0.1)





def main():

    rospy.init_node('ros_image_processor', anonymous=True)

    ipn = ImageProcessorNode("请处理图像并返回结果")

    t = threading.Thread(target=ipn.command_input_thread)

    t.daemon = True

    t.start()

    rospy.on_shutdown(lambda: cv2.destroyAllWindows())

    try:

        rospy.spin()

    except KeyboardInterrupt:

        pass

    finally:

        cv2.destroyAllWindows()





if __name__ == '__main__':

    main()

