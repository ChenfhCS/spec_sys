import zmq
import json
import torch
import base64
import threading
import numpy as np

from queue import Queue
from torch import Tensor
from transformers import AutoModelForCausalLM

from small_model_for_server_thread_three_model import max_fn, sample, norm_logits
# from server.server_thread_with_lock import ServerThread
from server.server_thread import ServerThread


# from server.server_threadpool import ServerThread
# torch.manual_seed(123)


class Server(ServerThread):

    def __init__(self):
        super().__init__("6006")
        # 定义大模型
        self.torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.large_model = AutoModelForCausalLM.from_pretrained(
            "/root/autodl-tmp/wanjie/data_and_model/model/huggyllama/llama-13b",
            torch_dtype=torch.float16).to(self.torch_device)
        self.large_model.eval()

        # 关键：定义一个信号量，最多允许 1 个线程同时进入
        self.model_call_sem = threading.Semaphore(value=1)
        self.lock = threading.Lock()

    @staticmethod
    def process_message_q_x(message: dict) -> tuple[Tensor, Tensor]:
        """将客户端发来的JSON数据解析成 Tensor"""
        # q_shape = tuple(message["q_shape"])
        # q_data = base64.b64decode(message["q_data"])
        # q_dtype = message["q_dtype"]
        # q_array = np.frombuffer(q_data, dtype=eval("np.{}".format(q_dtype))).reshape(q_shape)
        q_array_list = message["q_array_list"]

        x_shape = tuple(message["x_shape"])
        x_data = base64.b64decode(message["x_data"])
        x_dtype = message["x_dtype"]
        x_array = np.frombuffer(x_data, dtype=eval("np.{}".format(x_dtype))).reshape(x_shape)
        return q_array_list, torch.from_numpy(x_array.copy())

    def handle_client(self, client_id: bytes, queue: Queue, router_socket: zmq.Socket):
        while True:
            raw_data = queue.get()  # 阻塞等待消息
            message = json.loads(raw_data.decode())
            if message.get("stop") == "stop":
                print(client_id, message["stop"])
                # 主线程要求此线程结束（服务器整体要关闭 或 清理）
                # print(self.client_map)
                self._cleanup_client(client_id)
                # print(self.client_map)
                return

            # if message.get("signal") == "quit":
            #     # 收到客户端的退出请求，给它返回一个 quit_ack 后退出
            #     ack = {"signal": "quit_ack"}
            #     router_socket.send_multipart([client_id, json.dumps(ack).encode()])
            #     print(f"[Server] Thread for client {client_id} is closing...")
            #     return

            max_len = message["max_len"]
            gamma = message["gamma"]
            temperature = message["temperature"]
            top_k = message["top_k"]
            top_p = message["top_p"]
            p_plus = message["p_plus"]

            prefix = self.process_message(message).to("cuda:0")
            prefix_len = prefix.shape[1]
            T = prefix_len + max_len

            # current_spec_token = 0
            # current_acc_token = 0
            # cumulative_acc = []
            while prefix.shape[1] < T:
                prefix_len = prefix.shape[1]
                # 1. 等待客户端发送数据
                raw_data = queue.get()  # 阻塞等待消息
                message = json.loads(raw_data.decode())
                if message["signal"] == "always":
                    while prefix.shape[1] < T:
                        x = prefix
                        # 2. 准备回应的数据，完全由大模型生成
                        with self.model_call_sem:
                            p = self.large_model(x).logits
                        for i in range(p.shape[1]):
                            p[:, i, :] = norm_logits(p[:, i, :], temperature, top_k, top_p)
                        t = sample(p[:, -1, :])
                        prefix = torch.cat((prefix, t), dim=1)

                    # 发送prefix
                    prefix_array = prefix.cpu().numpy()[:, prefix_len:]
                    data = base64.b64encode(prefix_array).decode("ascii")
                    response = {
                        "shape": prefix_array.shape,
                        "dtype": str(prefix_array.dtype),
                        "data": data
                    }
                    with self.lock:
                        router_socket.send_multipart([client_id, json.dumps(response).encode()])
                    break
                elif message["signal"] == "immediate":
                    prefix_shape = tuple(message["prefix_shape"])
                    prefix_data = base64.b64decode(message["prefix_data"])
                    prefix_dtype = message["prefix_dtype"]
                    prefix_array = np.frombuffer(prefix_data, dtype=eval("np.{}".format(prefix_dtype))).reshape(
                        prefix_shape)
                    prefix_tensor = torch.from_numpy(prefix_array).to("cuda:0")
                    prefix = torch.cat([prefix, prefix_tensor], dim=1)
                else:
                    # print(message.keys())
                    q, x = self.process_message_q_x(message)
                    x = x.to("cuda:0")
                    # print(q, x.shape)
                    x = torch.cat([prefix, x], dim=1)

                    # 2. 准备回应的数据
                    with self.model_call_sem:
                        p = self.large_model(x).logits
                    for i in range(p.shape[1]):
                        p[:, i, :] = norm_logits(p[:, i, :], temperature, top_k, top_p)

                    # Determine the end position of the valid prefix
                    is_all_accept = True
                    n = prefix_len - 1

                    # current_spec_token += gamma
                    for i in range(gamma):
                        r = torch.rand(1, device="cpu")
                        # r = 0.4  # 0.1 or 0.2 or 0.3 or 0.4
                        j = x[:, prefix_len + i]
                        large_model_p = float(p[:, prefix_len + i - 1, j].cpu().detach().numpy()[0, 0])
                        small_model_q = q[i]
                        # for循环小模型的整个窗口，满足条件接收一个token，如果不满足，break跳出循环
                        # p_plus = 0.05 or 0.1 or 0.15 or 0.2
                        if r < min(1.0, (large_model_p + p_plus) / small_model_q):
                        # if True:
                            # if r < torch.min(torch.tensor([1.0], device=p.device),
                            #                  p[:, prefix_len + i - 1, j] / q[i]):
                            # Accept and update n
                            n += 1
                            # current_acc_token += 1
                            # if current_acc_token % 10 == 0:
                            #     cumulative_acc.append(round(current_acc_token / current_spec_token, 2))
                        else:
                            # Reject and sample correction
                            response = {
                                "q_n": n,
                            }
                            with self.lock:
                                router_socket.send_multipart([client_id, json.dumps(response).encode()])
                            # router_socket.send_multipart([client_id, json.dumps(response).encode()])

                            raw_data = queue.get()  # 阻塞等待消息
                            message = json.loads(raw_data.decode())
                            q_n = message["q_n"]
                            t = sample(max_fn(p[:, n, :] - torch.from_numpy(np.array(q_n)).to(p.device)))
                            # t = sample(p[:, n, :])
                            is_all_accept = False
                            break

                    prefix = x[:, :n + 1]

                    # If all tokens are accepted, sample from the last target logits
                    if is_all_accept:
                        t = sample(p[:, -1, :])
                    prefix = torch.cat((prefix, t), dim=1)

                    # 发送prefix
                    prefix_array = prefix.cpu().numpy()[:, prefix_len:]
                    data = base64.b64encode(prefix_array).decode("ascii")
                    response = {
                        "shape": prefix_array.shape,
                        "dtype": str(prefix_array.dtype),
                        "data": data,
                        # "large_model_p": large_model_p,
                        # "small_model_q": small_model_q,
                        # "r": float(r.numpy()[0])
                    }
                    with self.lock:
                        router_socket.send_multipart([client_id, json.dumps(response).encode()])
                    # router_socket.send_multipart([client_id, json.dumps(response).encode()])


def main():
    s = Server()
    s.run_server()


if __name__ == '__main__':
    main()
