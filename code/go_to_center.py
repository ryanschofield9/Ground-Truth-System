import numpy as np

def organize_dis (dis_readings):
    start = -1
    end = -1
    for idx, val in enumerate(dis_readings):
        if 150<val<400:
            if start == -1: 
                start = idx 
            else: 
                end = idx 
    
    dis_readings_organized = dis_readings[start:end]
    return dis_readings_organized, start

def determine_middle_center (dis_readings):
    dis_readings_organized, start = organize_dis(dis_readings) 
    if len(dis_readings_organized) == 0: 
        center_loc = -1 #no branch 
    else:
        center_loc = len(dis_readings_organized)/2 + start 
    return center_loc 
    
def determine_middle_low (dis_readings):
    dis_readings_organized, start = organize_dis(dis_readings)
    if len(dis_readings_organized) == 0: 
        low_loc = -1 #no branch 
    else:
        low_loc = np.argmin(dis_readings_organized)+start
    return low_loc

def determine_center_loc (tof1_dis_readings, tof2_dis_readings):
    loc_middle_tof1 = determine_middle_center(tof1_dis_readings) 
    loc_middle_tof2 = determine_middle_center(tof2_dis_readings)

    if loc_middle_tof1 == -1 or loc_middle_tof2 == -1:
        print("One or both tof sensors did not see a branch. Please try again")
        angle = 0 
    else: 
        #distance = (number of readings since start to middle / number of readings per second ) *speed
        num_readings_sec = 10 #rate is set to 0.1 
        speed = 0.1  # m/s
        dis_sensors = 0.0508 # meters
        dis_tof1 = (loc_middle_tof1/num_readings_sec) * speed # meters 
        dis_tof2 = (loc_middle_tof2/num_readings_sec) * speed # meters

        angle = np.arctan((dis_tof2-dis_tof1) / dis_sensors)
        
        #start_pos_y = known 
        #start_pos_y + (dis_tof1+dis_tof2 )/2 
        #x and z pos stay the same 

    return angle 

def main():
    '''
    #test 1
    tof1= [768,792,806,791,813,786,782,801,750,775,786,784,783,756,709,638,599,500,458,430,397,
            353,332,326,309,291,285,284,266,271,263,258,251,254,264,263,277,263,269,282,293,299,
            320,345,373,408,438,508,529,624,709,799,834,883,919,964,980,977,937,956,978,984,977,
            988,1001,967]
    tof2= [820,822,848,835,845,815,856,829,831,868,854,815,839,811,721,665,548,485,439,378,367,317,
            303,294,282,270,266,257,251,245,239,232,230,238,238,240,235,245,257,255,261,256,282,293,
            313,330,376,414,474,514,611,741,822,873,949,976,955,961,985,984,978,975,987,990,979,967]
    '''
    '''
    #test 2
    tof1 = [365,351,355,381,347,357,380,374,326,334,322,314,287,277,265,263,262,257,252,258,255,263,
            264,258,287,269,290,298,309,331,336,364,430,439,507,544,628,693,827,809,883,922,970,986,
            964,1000,1016,1005,1020,1026,1008,1025,1028,1025,1026,1015,1018,1013,1040,1025,1010,1029,
            1020,1018,1040,1020]
    tof2 =[345,330,348,342,345,334,334,340,340,302,297,272,266,250,255,248,234,238,237,233,235,235,
           240,247,244,260,258,271,269,301,290,315,364,362,429,481,531,672,770,790,896,914,984,976,
           989,1017,1040,1022,1027,1023,1023,1025,1024,1034,1025,1026,1019,1031,1029,1024,1025,1025,
           1024,1053,1041,1023]
    '''
    '''
    # test 3
    tof1 = [969,963,968,993,953,938,939,952,925,902,879,838,772,678,620,558,545,457,430,388,362,351,
            333,300,287,277,279,269,269,254,254,254,259,252,262,259,257,265,276,288,292,299,303,317,
            376,386,413,481,505,601,709,708,802,852,898,947,978,1001,984,1024,1022,1023,1024,1027,
            1014,1016,1027,1040,1041,1020,1054,1033,1028,1016,1019,1026,1040,1024,1022,1026,1016,
            1028,1027]
    
    tof2 = [979,986,979,995,971,1001,966,969,941,949,944,869,818,730,661,594,524,446,419,364,340,320,
            305,278,282,282,254,255,246,234,233,239,232,235,237,238,239,242,262,255,265,263,270,297,
            324,321,358,397,415,543,626,630,790,860,900,934,1005,1010,1013,1021,1024,1042,1035,1018,
            1028,1015,1042,1039,1035,1032,1029,1020,1025,1035,1037,1028,1042,1031,1030,1025,1007,
            1023,1034]
    '''

    tof1 = [1018,1014,1026,1033,1027,1022,1038,1040,1014,1038,1022,1030,1040,1023,1036,1051,1026,1035,
            1045,1028,1026,1037,1033,1041,1015,1021,1023,1024,1018,1012,1025,1025,1031,1037,1029,1015,
            1000,1018,1005,1017,1051,1002,1034,1020,1009,1019,1023,1026,1031,1027,1013,1011,1010,1023,
            1000,1021,1015,1049,1014,1032,1043,1005,1043,1023,1040,1007,1027,1030,1033,1033,1008,1015,
            1038,1024,1021,1026,1005,1017,1028,1016,1018,1016,1018]
    
    tof2 =[1043,1006,1034,1041,1026,1029,1053,1037,1028,1042,1045,1036,1017,1025,1026,1025,1028,1042,
           1025,1016,1032,1016,1024,1042,1016,1035,1018,1041,1036,1043,1038,1021,1016,1039,1030,1019,
           1042,1024,1029,1025,1030,1022,1055,1022,1015,1031,1025,1034,1046,1023,1025,1025,1017,1025,
           1022,1024,1055,1021,1041,1027,1034,1019,1037,1018,1028,1003,1027,1014,1037,1020,1022,1027,
           1049,1031,1033,1026,1040,1028,1029,1026,1033,1019,1010] 

    print(f"Middle Center of TOF1: {determine_middle_center(tof1)}")
    print(f"Middle Low of TOF1: {determine_middle_low(tof1)}")
    print(f"Middle Center of TOF2: {determine_middle_center(tof2)}")
    print(f"Middle Low of TOF2: {determine_middle_low(tof2)}")
    print(f"Angle: {determine_center_loc(tof1, tof2)} ")

main()