import rclpy 
from rclpy.node import Node 
from std_msgs.msg import Int64
from geometry_msgs.msg import TwistStamped, Vector3


class MoveArm(Node):
    def __init__(self):
        super().__init__('move_arm')
        # create publishers and subscripers (and timers as necessary )
        self.sub_tof = self.create_subscription(Int64, 'tof_fake', self.callback_tof, 10)
        self.pub = self.create_publisher(TwistStamped, '/servo_node/delta_twist_cmds', 10)
        self.pub_timer = self.create_timer(1/10, self.publish_twist)

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

    def publish_twist(self): 
        my_twist_linear = [0.0, 0.0, 0.0] 
        my_twist_angular = [0.0, 0.0, 0.0]

        # positive y makes it go down 
        # negative y makes it go up

        # sif the apple is centered and has been centered for about 1.5 seconds 
        if self.centered == False or self.count < 35: 
            # set twist messages to send
            if self.x_centered == 0: 
                    
                my_twist_linear[0]= 0.0
            
            elif self.x_centered == -1: 
                my_twist_linear[0] = self.speed 
            
            else: 
                my_twist_linear[0] = -1*self.speed 
                    
            if self.y_centered == 0: 
                my_twist_linear[1] = 0.0

            elif self.y_centered == -1:
                my_twist_linear[1] = -1 * self.speed 
            
            else: 
                my_twist_linear[1] = self.speed 
            
            if self.x_centered == 0 and self.y_centered == 0: 
                self.centered = True
                self.count = self.count + 1
            else: 
                self.centered = False 
                self.count = 0 
        # if the apple is centered and has been for about 1.5 seconds start to moving forward  
        else: 
            if self.tof >self.distance:
                my_twist_linear[2] = self.speed 
            
        cmd = TwistStamped()
        cmd.header.frame_id = 'tool0'
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.twist.linear = Vector3(x=my_twist_linear[0], y=my_twist_linear[1], z=my_twist_linear[2])
        cmd.twist.angular = Vector3(x=my_twist_angular[0], y=my_twist_angular[1], z=my_twist_angular[2])
        self.get_logger().info(f"Sending: linear: {cmd.twist.linear} angular: {cmd.twist.angular}")

        self.pub.publish(cmd)
    
    def callback_x (self, msg):
        self.x_centered = msg.data 
    
    def callback_y (self, msg):
        self.y_centered = msg.data
    
    def callback_timer(self):
        self.speed = self.get_parameter('speed').get_parameter_value().double_value
        self.distance = self.get_parameter('distance').get_parameter_value().integer_value
    
    def callback_tof (self, msg):
        self.tof = msg.data

def main(args=None):
    rclpy.init(args=args)
    move = MoveArm()
    rclpy.spin(move)
    rclpy.shutdown ()

if __name__ == '__main__':
   main()