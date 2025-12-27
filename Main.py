import argparse
import Task_Manager  

def main():
    parser = argparse.ArgumentParser(description="SpiderWeb Model Manager")
    # 修改点：去掉了 required=True，增加了 default='train'
    # 这样如果你不加参数直接运行，它就会默认执行训练模式
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'predict', 'optimize', 'invert'],
                        help="选择运行模式: train(训练), predict(预测), optimize(优化), invert(反求)")
    
    args = parser.parse_args()
    
    print(f"================ 正在启动模式: {args.mode} ================")
    
    if args.mode == 'train':
        Task_Manager.train_model()
    elif args.mode == 'predict':
        Task_Manager.run_prediction()
    elif args.mode == 'optimize':
        Task_Manager.run_optimization()
    elif args.mode == 'invert':
        Task_Manager.run_inversion()

if __name__ == "__main__":
    main()


# 例如，终端运行 D:/Anaconda3/envs/pytorch/python.exe c:/Deeplearning/NN/Main.py --mode optimize