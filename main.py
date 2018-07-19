from model import run
import sys

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: main.py <item|all> <n_epochs> <lr>\r\nItems are A,B,...")
        sys.exit(0)
    if sys.argv[1] == 'all':
        for item in 'ABCDEFGHI':
            print("Item(%s) ################################################"% item)
            run(item , int(sys.argv[2]), float(sys.argv[3]) )
    else:
        run(sys.argv[1] , int(sys.argv[2]), float(sys.argv[3]) )
