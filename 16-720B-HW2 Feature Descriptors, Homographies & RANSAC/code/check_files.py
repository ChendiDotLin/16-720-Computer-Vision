import os
andrew_id = 'XXX'


def check_file(file):
    if os.path.isfile(file):
        return True
    else:
        print('{} not found!'.format(file))
        return False
    

if ( check_file('../'+andrew_id+'/code/BRIEF.py') and \
     check_file('../'+andrew_id+'/code/keypointDetect.py') and \
     check_file('../'+andrew_id+'/code/panoramas.py') and \
     check_file('../'+andrew_id+'/code/planarH.py') and \
     check_file('../'+andrew_id+'/results/6_1.npy') and \
     check_file('../'+andrew_id+'/results/testPattern.npy') and \
     check_file('../'+andrew_id+'/'+andrew_id+'_hw2.pdf') ):
    print('file check passed!')
else:
    print('file check failed!')

#modify file name according to final naming policy
#you should also include files for extra credits if you are doing them (this check file does not check for them)
#images should be be included in the report
