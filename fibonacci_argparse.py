import argparse

def fib(n):
    a,b = 0,1
    for i in range(n):
        a,b = b,a+b
    return a

# Main function
def Main():
    # Creating parser
    parser = argparse.ArgumentParser()

    # Adding a group for MUTUALLY EXCLUSIVE ARGS
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-v','--verbose',action="store_true")
    group.add_argument('-q','--quiet',action="store_true")

    # Add positional argument
    parser.add_argument('num', help="The fibonacci number you wish to calculate", type=int)
    # Modify parser by adding optional args - both short and long version and defining the action
    parser.add_argument('-o','--output', help="Output result to a file.",action="store_true")

    # grab args from command line
    args = parser.parse_args()
    # calculating results - num is the argument specified
    result = fib(args.num)

    # USING MUTUALLY EXCLUSIVE GROUPS
    if args.verbose:
        # print result
        print(f'The {str(args.num)}th fib number is {str(result)}') 
    elif args.quiet:
        print(result)
    else:
        print("Fib("+str(args.num)+") = "+str(result))

    # FOR OPTIONAL ARGUMENMT:  If output is stored as True, this code stores the result into a file
    if args.output:
        f = open('fibonacci.txt','a')
        f.write(str(result)+'\n')

if __name__=='__main__':
    Main()