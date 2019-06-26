# # Short Arguments
# import argparse
# parser = argparse.ArgumentParser(
#     description="Short Sample App")
# parser.add_argument('-a', action="store_true", default = False)
# parser.add_argument('-b', action="store", dest="b")
# parser.add_argument('-c', action="store", dest="c", type=int)
# print(parser.parse_args(['-a','-bval','-c','3']))

# # Long Arguments
# import argparse
# parser = argparse.ArgumentParser(description='Example with long option names',)
# parser.add_argument('--noarg', action="store_true", default=False)
# parser.add_argument('--witharg', action="store",dest="witharg")
# parser.add_argument('--witharg2', action="store", dest="witharg2", type=int)
# print(parser.parse_args(['--noarg', '--witharg', 'val', '--witharg2=3']))


# # Test handling required arguments
# import argparse
# parser = argparse.ArgumentParser(
#     description='Example with nonoptional arguments',
# )
# parser.add_argument('count', action="store", type=int)
# parser.add_argument('units', action="store")
# print(parser.parse_args())
