from model import run
import sys

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: main.py <n_epochs> <item|all>\r\nItems are A,B,...")
        sys.exit(0)
    if sys.argv[2] == 'all':
        for item in 'ABCDEFGHI':
            print("Item(%s) ################################################"% item)
            run(int(sys.argv[1]), item)
    else:
        run(int(sys.argv[1]), sys.argv[2])
