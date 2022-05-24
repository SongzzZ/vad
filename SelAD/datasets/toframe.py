# coding: UTF-8
import os
import cv2

# 遍历文件夹
def walkFile(file):
    for root, dirs, files in os.walk(file):
    # root 表示当前正在访问的文件夹路径
    # dirs 表示该文件夹下的子目录名list
    # files 表示该文件夹下的文件list
    # 遍历文件
        for f in files:
            
            print(os.path.join(root, f).split('/')[2].split('.')[0])     # save name
    # 遍历所有的文件夹
        for d in dirs:
            print(os.path.join(root, d))
 
          
def main(save2path):  
    # walkFile("./videos/")
    for root, dirs, files in os.walk("./videos/"):
        for vid in files:
            if ('.avi' in vid):
                path = os.path.join(root, vid)
                save_path = save2path + str(path.split('/')[2].split('.')[0]) + '/'           
                # create save_dir
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                
                print(path)
                video = cv2.VideoCapture(path)
                video_fps = int(video.get(cv2.CAP_PROP_FPS))            # get fps

                print('Saving frames {}:'.format(vid))
               
               # processing videos
                cur_frame = 0
                success, frame = video.read()
                while success:
                    # print(frame)
                    # print(save_path + path.split('/')[3].split('.')[0] + '/' + str(cur_frame) + '.jpg')
                    addr = save_path + str("%03d" % cur_frame) + '.jpg'
                    cv2.imwrite(addr, frame)
                    cur_frame = cur_frame + 1
                    success, frame = video.read()
                
                video.release()
                print('{} complete!'.format(vid))
    
if __name__ == '__main__':
    
    save2path = "./frames/"
    main(save2path)