import numpy
from submodopt.submodopt import SubmodularOpt
from time import time
import numpy as np
sents =[]
source = 'what is the easiest way to earn money online'
import pdb

# sents.append('what is best way to make money online' )
# sents.append('what should i do to make money online' )
# sents.append('what should i do to earn money online' )
# sents.append('what is the easiest way to make money online' )
# sents.append('what is the easiest way to earn money online' )
# sents.append('what s the easiest way to make money online' )
# sents.append('what s the easiest way to earn money online' )
# sents.append('what should i do to make money online online' )
# sents.append('what is the best way to make money online' )
# sents.append('what is the easiest way to make money online online' )

source ='where can i get affordable package in sydney for floor tiles ?'
sents =np.array(['where can i found wide variety of ceramic tiles in sydney ?',
       'where can i get huge selection of floor tiles in sydney ?',
       'where can i get designer collection of floor tiles in sydney ?',
       'where can i found large collections for ceramic and porcelain tiles in sydney ?',
       'where can i get amazing collection of floor tiles in sydney ?',
       'where can i get wonderful collection of floor tiles in sydney ?',
       'where can i get affordable types of stylish ceramic floor tiles in sydney ?',
       'where can i get affordable collection of stylish floor tiles in sydney ?',
       'where can i get designer collection of affordable floor tiles in sydney ?',
       'where can i found large collections of ceramic and porcelain wall in sydney ?',
       'where can i found quality floor tile collections in sydney ?',
       'where can i found variety of ceramic tiles in sydney ?',
       'where can i get wide variety of floor tiles in sydney ?',
       'where can i get wonderful collection of stylish floor tiles in sydney ?',
       'where can i get wonderful package in sydney for floor tiles ?',
       'where can i get wide variety of ceramic tiles in sydney ?',
       'where can i get wonderful shopping experience in sydney for floor tiles ?',
       'where can i found quality floor tiles in sydney ?',
       'where can i get affordable types of floor tiles in sydney ?',
       'where can i found wide variety of floor tiles in sydney ?',
       'where can i get wonderful types of floor tiles in sydney ?',
       'where can i get designer collection of ceramic tiles in sydney ?',
       'where can i found very durable outdoor tiles in sydney ?',
       'where can i get affordable collection of floor tiles in sydney ?',
       'where can i get affordable package in sydney for floor tiles ?',
       'where can i found variety of floor tiles in sydney ?',
       'where can i get wonderful floor tiles in sydney ?',
       'where can i get various quality floor tile collections in sydney ?',
       'where can i get variety of ceramic tiles in sydney ?',
       'where can i get large package in sydney for floor tiles ?',
       'where can i get wonderful types of floor tile services in sydney ?',
       'where can i get wonderful range of floor tiles in sydney ?',
       'where can i found quality floor tile collection in sydney ?',
       'where can i get wide selection of floor tiles in sydney ?',
       'where can i get affordable package in sydney for bathroom floor tiles ?',
       'where can i get designer floor tiles in sydney ?',
       'where can i get wide selection of floor tiles in sydney',
       'where can i get huge selection of floor tiles in sydney',
       'where can i get wide selection of floor tiles in tiles ?',
       'where can i found designer collection in sydney ?',
       'where can i get designer collection of floor tiles in sydney',
       'where can i get amazing floor tiles in sydney ?',
       'where can i found wide variety of ceramic tiles in sydney',
       'where can i get wonderful types of floor tiles in sydney',
       'where can i get affordable types of floor tiles in sydney ? ?',
       'where can i get wide selection of floor tiles in sydney ? ?',
       'where can i get designer collection of affordable floor tiles in sydney',
       'where can i get affordable collection of stylish floor tiles in sydney',
       'where can i get wonderful shopping experience in sydney for floor tiles',
       'where can i get wonderful collection of stylish floor tiles in sydney'])

st_time = time()
# pdb.set_trace()

subopt= SubmodularOpt(sents, source)
subopt.initialize_function(0.5)
selec_sents= subopt.maximize_func(10)
etime = (time() - st_time)/60.0
print('Time Taken : {}'.format(etime))


print('original sentences')
print('--------------------------------------------')
for sent in sents:
    print(sent)
print('--------------------------------------------')
print('selected sentences')
print('--------------------------------------------')
for sent in selec_sents:
    print(sent)

print('--------------------------------------------')
