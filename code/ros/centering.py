import rclpy 
from rclpy.node import Node 
from std_msgs.msg import Int64
from geometry_msgs.msg import TwistStamped, Vector3
import time 


class MoveArm(Node):
    def __init__(self):
        super().__init__('move_arm')
        # create publishers and subscripers (and timers as necessary )
        self.sub_tof1 = self.create_subscription(Int64, 'tof1', self.callback_tof1, 10)
        self.sub_tof2 = self.create_subscription(Int64, 'tof2', self.callback_tof, 10)
        self.pub_vel_commands = self.create_publisher(TwistStamped, '/servo_node/delta_twist_cmds', 10)
        self.pub_timer = self.create_timer(1/10, self.publish_twist)
        #add a publisher that will be able to publish position (x,y,z,rotation) or joint 
        #self.pub_joint_commands= 
        #add a subscriber that will be able to get position location (proferably x,y,z, rotation tool endeffector)
        #self.sub_loc = 

        #Creating parameters 
        self.declare_parameter('speed',0.1)
        self.declare_parameter('distance', 20)
        self.timer_2 = self.create_timer(1, self.callback_timer)
        
        #variables needed 
        self.centered = False
        self.speed = 0.1
        self.distance = 20
        self.y_centered = 0 
        self.x_centered = 0 
        self.count = 0 

        self.start_time = time.time() 
        self.tof_collected = False
        self.move_up_collect = 5 #seconds needed to move up and collect tof data 
        self.tof1_readings = []
        self.tof2_readings = []
        self.lowest_reading_tof1 = 500 #start with value that can not be saved 
        self.lowest_reading_tof2 = 500 #start with value that can not be saved 
        self.cal_angle_done = False

    def publish_twist(self): 
        my_twist_linear = [0.0, 0.0, 0.0] 
        my_twist_angular = [0.0, 0.0, 0.0]

        if self.tof_collected == False: 
            # if tof data has not been collected 
            now = time.time()
            if (now - self.start_time ) < self.move_up_collect:
                #if it has not been the seconds needed to move up and collect tof data, keep moving up at 0.1 m/s
                my_twist_linear[1]=  -0.1 #moving up at -0.1 m/s 
            else:
                self.tof_collected = True
        else: 
            if self.calc_angle_done ==False:
                self.calculate_angle()
            #Add here the function that will find the position 
            #Add here the publisher that will send the robot to the saved position 

        cmd = TwistStamped()
        cmd.header.frame_id = 'tool0'
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.twist.linear = Vector3(x=my_twist_linear[0], y=my_twist_linear[1], z=my_twist_linear[2])
        cmd.twist.angular = Vector3(x=my_twist_angular[0], y=my_twist_angular[1], z=my_twist_angular[2])
        self.get_logger().info(f"Sending: linear: {cmd.twist.linear} angular: {cmd.twist.angular}")

        self.pub.publish(cmd)
    
    
    def callback_timer(self):
        self.speed = self.get_parameter('speed').get_parameter_value().double_value
        self.distance = self.get_parameter('distance').get_parameter_value().integer_value
    
    def callback_tof1 (self, msg):
        now = time.time()
        if (now - self.start_time ) < self.move_up_collect:
             self.tof1_readings.append(msg.data)
             #Add here if between 150 and 400 save the lowest joint pos 
             #if 150 < msg.data < 400:
                #if msg.data < self.lowest_reading_tof1:
                    #self.lowest_pos_tof1 = find a way to save position 
    
    def callback_tof2(self, msg):
        now = time.time()
        if (now - self.start_time ) < self.move_up_collect:
             self.tof2_readings.append(msg.data)
             #Add here if between 150 and 400 save the lowest joint pos 
             #if 150 < msg.data < 400:
                #if msg.data < self.lowest_reading_tof2:
                    #self.lowest_pos_tof2= find a way to save position 
    
    def calculate_angle(self):
        print(self.tof1_readings)
        print(self.tof2_readings)


def main(args=None):
    rclpy.init(args=args)
    move = MoveArm()
    rclpy.spin(move)
    rclpy.shutdown ()

if __name__ == '__main__':
   main()