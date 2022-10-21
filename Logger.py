"""
2CET-GAN
Copyright (c) 2022-present, Xiaohang Hu.
This work is licensed under the MIT License.
"""

import time


class LoggerOutStdout:
    def out(self, log):
        print(log)

    def close(self):
        None


class LoggerOutFile:
    def __init__(self, file_name):
        self.log_file = open(file_name, 'a+', encoding='UTF8', newline='')
        print(f"Log to {file_name}.")

    def out(self, log):
        log = log + "\r\n"
        self.log_file.write(log)
        self.log_file.flush()

    def close(self):
        self.log_file.close()


class Logger:

    def __init__(self, outs, log_dir, file_name):
        outers = []
        for out in outs:
            if out == "stdout":
                outers.append(LoggerOutStdout())
            elif out == "file":
                file_name = f"{log_dir}/{file_name}"
                outers.append(LoggerOutFile(file_name))
            else:
                raise Exception(f"Unknow out type :{out}!")
        self.outers = outers

    def log(self, log):
        log_time = time.strftime("[%d/%m/%Y %H:%M:%S]")
        log = log_time + " " + log
        for outer in self.outers:
            outer.out(log)

    def close(self):
        for outer in self.outers:
            outer.close()
