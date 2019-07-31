'''
Copyright@ Qiao-Mu(Albert) Ren. 
All Rights Reserved.
This is the code of HairNet.
Last modified by rqm @22:40, July 31, 2019
'''
import argparse
from model import train, test

parser = argparse.ArgumentParser(description='This is the implementation of HairNet by Qiao-Mu(Albert) Ren using Pytorch.')

parser.add_argument('--mode', type = str, default='train')
parser.add_argument('--path', type = str, default=r'F:\QiaomuRen_Data\HairNet_training_data')
parser.add_argument('--weight', type = str, default='/home/albertren/Workspace/HairNet/HairNet-ren/weight/000001_weight.pt')
args = parser.parse_args()

def main():
    if args.mode == 'train':
        print(args.path)
        train(args.path)
    if args.mode == 'test':
        test(args.path, args.weight)

if __name__ == '__main__':
    main()


