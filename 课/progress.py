# 实现一个通用进度条，进度条的样式自定义
# 函数入参要求：
# 1. 进度条每进一格所需要的秒数 seconds
# 2. 进度条完成之后执行的函数 callback_function
# 函数功能要求：
# 1. 实现每seconds秒进度条进一步
# 2. 进度条达到100%之后，调用`callback_function`
# 函数返回要求：
# None
# 备注：实际调用时，callback_function可以自己定义，至少定义两种callback_function，实际调用时也分别调用一次（传入不同的callback_function）
import time
def progress_bar (seconds,callback_function):
    total_steps = 100    #进度条的总步长
    for step in range (total_steps + 1):
        percent = step  / total_steps * 100
        bar = '■' *step + ' ' * (total_steps - step)
        print (f"\r[{bar}]{percent:0f}%",end="")
        time.sleep (seconds)   #每步等待指定的秒数
    
    callback_function()  #调用回调的函数


#定义两个回调函数
def callback_function_1():
    print ("回调函数1已运行结束")


def callback_function_2():
    print ("回调函数2已运行结束")



if __name__ == "__main__":
    progress_bar (0.01,callback_function_1)
    progress_bar (0.01,callback_function_1)


