# with open("list_attr_celeba.txt") as xh:
#     with open('identity_CelebA.txt') as yh:
#         with open('list_eval_partition.txt') as zh:
#             with open("merge.txt","w") as mh:
#                   xlines = xh.readlines()
#                   ylines = yh.readlines()
#                   zlines = zh.readlines()
    
#                   mh.write(xlines[0])
#                   mh.write(xlines[1])
#                   for i in range(len(xlines)-2):
#                       line = xlines[i+2].strip() + ' ' + str(ylines[i].split()[1]) + ' ' + str(zlines[i].split()[1]) + "\n"
#                       mh.write(line)

with open("label.txt") as rh:
    lines = rh.readlines()
    print(len(lines))
    # max_id = -1
    # for line in lines:
    #     line = line.split()
    #     max_id = max(max_id, int(line[-1]))
    # print(max_id)
        