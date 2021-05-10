def DT_calculator(Days2run,DT,NDEFHIS,NHIS):
    '''
    Days2run = unit : (day/int) --> Days2run := Days2run*(60*60*24)
    DT       = unit : (hour/float) --> DT := DT*3600
    NDEFHIS  = unit : (N/int) --> NDEFHIS := NDEFHIS*DT
    NHIS     = unit : (N/int) --> NHIS := NHIS*3600

    INFO :

    '''
    Days2run_ = Days2run*(60*60*24)
    DT_ = DT*3600
    NTIMES_ = Days2run_/DT_
    NDEFHIS_ = NDEFHIS*DT_    
    NHIS_ = NHIS*3600

    if  str(NHIS_%DT_)[-2:] != '.0':
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('!!!Warning : NHIS%DT != int!!! ')
        print('!!!NHIS%DT : ',str(NHIS_%DT_)+'!')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('DT : '+str(DT_))
    print('NTIMES : '+str(NTIMES_))
    print('NHIS : '+str(NHIS_) )
    print('NDFHIS : '+str(NDEFHIS_) )



