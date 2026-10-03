#!/usr/bin/env python3
"""TCEI 视觉小窗：只读旁观件，实时显示带 bbox 的标注相机流。

与核对面板（desktop_review.py）同源的取流方式，但只做一件事：
显示 /tcei/annotated_image（感知侧已叠好每类一色 bbox）+ 候选类别清单。
无命令发布、无证据职责，随 run 的 --viewer 开关拉起，关窗即退出。
"""
import base64
import io
import json
import tkinter as tk

import cv2
import rospy
from cv_bridge import CvBridge
from PIL import Image as PILImage
from sensor_msgs.msg import Image as RosImage
from std_msgs.msg import String

WIDTH, HEIGHT = 800, 450
TICK_MS = 120


class Viewer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('TCEI 视觉 · bbox（只读）')
        self.status = tk.Label(self.root, text='等待相机与候选…', font=('TkDefaultFont', 11),
                               anchor='w', justify='left', bg='#222', fg='#eee')
        self.status.pack(fill='both')
        self.picture_holder = tk.Label(self.root)
        self.picture_holder.pack()
        self.photo = None
        self.latest = None
        self.candidates = []
        self.bridge = CvBridge()
        rospy.init_node('bbox_window', anonymous=True, disable_signals=True)
        rospy.Subscriber('/tcei/annotated_image', RosImage, self.on_image, queue_size=1, buff_size=2 ** 24)
        rospy.Subscriber('/tcei/candidates', String, self.on_candidates, queue_size=1)
        self.root.after(TICK_MS, self.tick)

    def on_image(self, msg):
        try:
            array = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            picture = PILImage.fromarray(cv2.cvtColor(array, cv2.COLOR_BGR2RGB))
            picture.thumbnail((WIDTH, HEIGHT), PILImage.LANCZOS)
            self.latest = picture
        except Exception:
            pass

    def on_candidates(self, msg):
        try:
            self.candidates = json.loads(msg.data).get('candidates', [])
        except Exception:
            pass

    def tick(self):
        if self.latest is not None:
            buf = io.BytesIO()
            self.latest.save(buf, format='GIF')
            self.photo = tk.PhotoImage(data=base64.b64encode(buf.getvalue()).decode('ascii'))
            self.picture_holder.configure(image=self.photo)
        names = '  '.join('#%s %s' % (c.get('id'), c.get('class')) for c in self.candidates)
        self.status.configure(text=names or '等待候选…')
        self.root.after(TICK_MS, self.tick)

    def run(self):
        self.root.mainloop()


def main():
    Viewer().run()


if __name__ == '__main__':
    main()
