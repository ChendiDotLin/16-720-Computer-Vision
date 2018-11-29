import os
andrew_id = 'XXX'


if ( os.path.isfile('../'+andrew_id+'/' + andrew_id + '/visual_words.py') and \
os.path.isfile('../'+andrew_id+'/' + andrew_id +'/visual_recog.py') and \
os.path.isfile('../'+andrew_id+'/' + andrew_id +'/network_layers.py') and \
os.path.isfile('../'+andrew_id+'/' + andrew_id +'/deep_recog.py') and \
os.path.isfile('../'+andrew_id+'/' + andrew_id +'/util.py') and \
os.path.isfile('../'+andrew_id+'/' + andrew_id +'/main.py') and \
os.path.isfile('../'+andrew_id+'/'+andrew_id+'_hw1.pdf') ):
    print('file check passed!')
else:
    print('file check failed!')

#modify file name according to final naming policy
#images should be included in the report
