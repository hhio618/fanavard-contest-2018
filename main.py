from model import run
import sys

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: main.py <item|all>\r\nItems are A,B,...")
        sys.exit(0)
    if sys.argv[1] == 'all':
        for item in 'ABCDEFGHI':
            print("Item(%s) ################################################"% item)
            run(item)
    else:
        run(item)
