import time
import ast 
import socket
import textwrap


class Robot:
    def __init__(self, robot_ip):
        self.ip = robot_ip
        #self.port = robot_port
        self.sock = None

    def connect(self, robot_port):
        try:
            self.sock = socket.create_connection((self.ip, robot_port), timeout=5)
            print(f"\nConnected to robot at {self.ip}:{robot_port}")
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
           
    def send_command(self, command: str, robot_port):
        if self.connect(robot_port):
            try:
                print("Sending command to robot...")
                self.sock.sendall(command.encode())
                #self.sock.close()
                return True
            except Exception as e:
                print(f"Failed to send command: {e}")
                return False
            
            finally:
                self.sock.close()
                print("Socket closed.")
        else:
            return None
    
    def receive_response(self, pc_ip, pc_port):
        timeout = 5
        """Set up server socket and wait for response from robot."""
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
            server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_sock.bind((pc_ip, pc_port))
            server_sock.listen(1)
            server_sock.settimeout(timeout)
            print(f"PC listening on {pc_ip}:{pc_port} for robot response...")
        
            try:
                conn, addr = server_sock.accept()
                print(f"\nPC sucessfully established connection with {addr}")
                with conn:
                    #print(f"Data received from robot: {addr}")
                    data = conn.recv(1024).decode()
                    return data
                    # print(f"Received data: {pose_data}")
            except socket.timeout:
                print("Timeout waiting for robot response.")
                return None
    
    def receive_response_has_reached(self, pc_ip, pc_port):
        timeout = 5
        
        """Set up server socket and continuously receive responses from robot."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
            server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_sock.bind((pc_ip, pc_port))
            server_sock.listen(1)
            server_sock.settimeout(timeout)
            print(f"\nPC listening on {pc_ip}:{pc_port} for robot responses...")

            try:
                conn, addr = server_sock.accept()
                print(f"PC sucessfully established connection with {addr}")
                with conn:
                    while True:
                        data = conn.recv(1024)
                        decoded = data.decode().strip()
                        print(f"Received data: {decoded}")
                        if "REACHED" in decoded:
                            return
                        time.sleep(0.5)
            except Exception as e:
                print(f"Error: {e}")
                    
                
                
    def movel(self, p, a, v, pc_ip, pc_port, robot_port):
        script = textwrap.dedent(
            f'''
            def test():
            
                def angle_diff(a, b):
                    return abs((a - b + 3.14159) % (2 * 3.14159) - 3.14159)

                def pose_diff(p1, p2):
                    dx = p1[0] - p2[0]
                    dy = p1[1] - p2[1]
                    dz = p1[2] - p2[2]
                    dpos = sqrt(dx*dx + dy*dy + dz*dz)
                    drx = angle_diff(p1[3], p2[3])
                    dry = angle_diff(p1[4], p2[4])
                    drz = angle_diff(p1[5], p2[5])
                    return [dpos, drx, dry, drz]
                
                def run_movel(target_pose, a, v):
                    movel(target_pose, a=a, v=v)

                socket_open("{pc_ip}", {pc_port})
                sleep(0.1)

                target, a, v = {p}, {a}, {v}
                thread_handler = start_thread(run_movel, (target, a, v))

                reached = False
                last_pose = get_actual_tcp_pose()

                pos_tol = 0.003  # 3mm
                rot_tol = 0.01   # 0.01 rad ~0.57 degrees
                
                while not reached:
                    sleep(0.5)
                    current_pose = get_actual_tcp_pose()
                    pos_diff_target, rotx_diff_target, roty_diff_target, rotz_diff_target = pose_diff(current_pose, target)
                    #pos_diff_last, rotx_diff_last, roty_diff_last, rotz_diff_last = pose_diff(current_pose, last_pose)

                    if pos_diff_target < pos_tol and rotx_diff_target < rot_tol and roty_diff_target < rot_tol and rotz_diff_target < rot_tol:
                        reached = True
                        socket_send_string("REACHED")
                    else:
                        socket_send_string("MOVING")
                        last_pose = current_pose

                socket_close()
            end'''
        )
        if self.send_command(script, robot_port):
            self.receive_response_has_reached(pc_ip, pc_port) #blocking
        
    def get_tcp_pose(self, pc_ip, pc_port, robot_port):
        """High level method to get TCP pose from robot."""
        script = textwrap.dedent(
            f'''
            def test():
                sleep(0.5) # adjust 0.5 or 1, longer delay for the robot to send back 
                socket_open("{pc_ip}", {pc_port})
                socket_send_string(get_actual_tcp_pose())
                socket_close()
            end'''
        )
        i = 0
        while(i<5):
            self.send_command(script, robot_port)                
            data = self.receive_response(pc_ip, pc_port)
            if data:
                tcp_data = ast.literal_eval(data.strip())
                print(f"Received tcp pose: {tcp_data}")
                return tcp_data
            else:
                print(f"ERROR  tcp pose = NONE")
                i+=1
            # else:
            #     return None
        print("max tries reached")
        return None
            
        
if __name__ == "__main__":

    robot_ip = '192.168.1.200'
    robot_port = 30001
    robot_port_pose = 30001
    pc_ip = '192.168.1.1' 
    pc_port = 30005
    robot = Robot(robot_ip)
    
    # # get tcp_data
    tcp_data = robot.get_tcp_pose(pc_ip, pc_port, robot_port_pose)
    if not (tcp_data is None):
        print(f'{tcp_data} \n')
    
 
    p_end = [-0.481263, -0.098807, 0.359674, 2.985337, 0.0537, -1.557183]
    p_start = [-0.242589, 0.119458, 0.329942, 3.122316, 0.037984, -2.477841]
    a, v = 0.05, 0.05
    
    robot.movel(p_start, a, v, pc_ip, pc_port, robot_port)
    robot.movel(p_end, a, v, pc_ip, pc_port, robot_port)
    
    

        
        
    
    
    

 
