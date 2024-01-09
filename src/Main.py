from MINDG import run
import torch

if __name__ == '__main__':
    run('DAVIS', phase="train",batch_size=32,epochs=5,learning_rate=5e-4,lr_step_size=10,seed_id=10,device=torch.device('cpu'))
    run('DAVIS')
    # run('BindingDB_Kd')
    # run('KIBA')
    # run('BindingDB_IC50')
    # run('BindingDB_Ki')