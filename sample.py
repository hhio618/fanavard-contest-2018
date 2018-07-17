file_names = ['A_book.csv', 'A_ticker.csv', 'A_trades.csv','B_book.csv', 'B_ticker.csv', 'B_trades.csv','C_book.csv', 'C_ticker.csv', 'C_trades.csv','D_book.csv', 'D_ticker.csv', 'D_trades.csv','E_book.csv', 'E_ticker.csv', 'E_trades.csv','F_book.csv', 'F_ticker.csv', 'F_trades.csv','G_book.csv', 'G_ticker.csv', 'G_trades.csv', 'H_book.csv', 'H_ticker.csv', 'H_trades.csv','I_book.csv', 'I_ticker.csv', 'I_trades.csv']

f = open('sample.txt', 'w')

for name in file_names:
    inp = open('data/' + name)
    if 'book' in name:
        f.write(' '.join(inp.readline().strip().split(',')))
    else:
        f.write(' '.join(inp.readline().strip().split(',')[1:]))

f.close()

for line in open('sample.txt'):
    print(len(line.strip().split()))