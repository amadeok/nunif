import tkinter as tk
import requests
import threading
import time

class SeekBarApp:
    def __init__(self, root=None, port=8123, get_info_fun=None, seek_fun=None, seek_relative_fun=None):
        
        self.root = tk.Tk()if not root else root
        
        self.port = port
        self.root.overrideredirect(True)
        self.root.attributes('-topmost', True)
        
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        self.window_width = self.screen_width 
        self.window_height = 60
        self.root.geometry(f"{self.window_width}x{self.window_height}+0+0")
        
        self.root.attributes('-alpha', 0.0)  # Start hidden
        self.hidden = True
        self.detection_height = self.window_height+20  # Area height to detect cursor
        self.get_info_fun = get_info_fun
        self.seek_fun = seek_fun
        self.seek_relative_fun = seek_relative_fun
        # Option 2: Use grid for more precise control
        main_frame = tk.Frame(root, bg="green")
        main_frame.pack(side="left", expand=True, fill='x', padx=0, pady=5)
        
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)
        
        containers :list[tk.Frame] = [None, None]
        dummy_color = "#2c2c2c"
        proportion = (1,20)
        for x in range(2):
            containers[x] = tk.Frame(main_frame, bg="blue")
            containers[x].grid(row=0, column=x, sticky="nsew")#(side="left", expand=True, fill='x', padx=2, pady=5)
            # containers[x].pack(side="left", expand=True, fill='x', padx=2, pady=5)
            containers[x].grid_columnconfigure(0, weight=proportion[0])
            containers[x].grid_columnconfigure(1, weight=proportion[1])
            containers[x].grid_columnconfigure(2, weight=proportion[0])
            # containers[x].grid_rowconfigure(0, weight=1)  # Add this line
            
            dummy_widget = tk.Frame(containers[x],  bg=dummy_color)# bg='#4d4d4d')
            dummy_widget.grid(row=0, column=0, sticky='nsew')
            
            if x == 0:
                self.seek_var = tk.DoubleVar()

                self.seek_scale = tk.Scale(
                    containers[x], from_=0, to=1000, orient='horizontal',
                    showvalue=False, sliderlength=20, width=self.window_height-10,
                    troughcolor='#4d4d4d', bg='#333333', fg='white',
                    variable=self.seek_var, highlightthickness=0
                )      
                self.seek_scale.grid(row=0, column=1, sticky='nsew', padx=(0, 0))
            else:
                dummy_widget_ = tk.Frame(containers[x],  bg='#4d4d4d', height=self.window_height-10)# bg='#4d4d4d')
                dummy_widget_.grid(row=0, column=1, sticky='nsew')   

            dummy_widget2 = tk.Frame(containers[x],  bg=dummy_color)# bg='#4d4d4d')
            dummy_widget2.grid(row=0, column=2, sticky='nsew')
        
        def update(e):
            def blink(color="#999999"):
                self.seek_scale.config( bg=color, troughcolor=color)
                self.root.after(500, lambda: self.seek_scale.config( bg='#333333',troughcolor="#4d4d4d"))
                
            def task():
                
                x = self.root.winfo_pointerx()
                # val = x / self.screen_width
                factor = proportion[0] / proportion[1]
                padx = self.screen_width*factor
                if x < padx:
                    print("seek back")
                    if self.seek_relative_fun: self.seek_relative_fun(-15)
                    else: print("relative seek not implemented")
                    blink("red")
                elif x > self.screen_width-padx:
                    print("seek forward")
                    if self.seek_relative_fun: self.seek_relative_fun(15)
                    else: print("relative seek not implemented")
                    blink("blue")
                else:
                    # xx = x+padx
                    val = (x-padx)*(1+(factor*2)) / self.root.winfo_width() 
                    # val = e.x / root.winfo_width() 
                    print("event", val)
                    blink()
                    self.on_slider_interaction(round(val*100))
            threading.Thread(target=task, daemon=True).start()

        self.root.bind("<Button-1>",  update)

        self.last_position = 0
        self.mouse_in_detection_area = False
        self.updating_from_server = False

        self.root.after(1, self.check_mouse_position)

        self.update_playback_time()
        if not root:
            self.root.mainloop()
            # threading.Thread(target=lambda: self.root.mainloop() , daemon=True).start()

    
    def fetch_playback_time(self):
        
        def do(data):
            self.current_playback_time  = data.get("playback_time", 1)
            self.duration  = data.get("duration", 1)
            self.perc_position = round( (self.current_playback_time / self.duration)*1000)
            # Update duration if available; adjust key as per your API
            self.duration = data.get("duration", 100)
            self.updating_from_server = True
            # Update the seek bar range and position
            # self.seek_bar.config(to=self.duration)
            # self.seek_scale.set(self.perc_position)
            self.seek_var.set(self.perc_position)
            self.updating_from_server = False
        try:
            if self.get_info_fun:
                data = self.get_info_fun()
                do(data)
            else:
                response = requests.get(f"http://localhost:{self.port}/api/info")
                if response.status_code == 200:
                    data = response.json()
                    do(data)


        except Exception as e:
            print("error", e)
        except requests.exceptions.RequestException as e:
            print(f"Error fetching playback time: {e}")
        return False
    
    
    def update_playback_time(self):
        """Update the seek bar position from server data."""       
        thread = threading.Thread(target=self.fetch_playback_time, daemon=True)
        thread.start()
        self.root.after(1000, self.update_playback_time)

    def on_slider_interaction(self, value):
        if self.updating_from_server:
            return
        if self.seek_fun:
            self.seek_fun(value)
        else:
            try:
                payload = {"position": value}
                response = requests.post(
                    "http://localhost:8123/api/seek",
                    json=payload
                )
                if response.status_code != 200:
                    print(f"Failed to update playback time: {response.text}")
            except requests.exceptions.RequestException as e:
                print(f"Error updating playback time: {e}")
            

    def check_mouse_position(self):
        """Check mouse position and show/hide window"""
        if 1:
            try:
                x = self.root.winfo_pointerx()
                y = self.root.winfo_pointery()
                
                in_detection_area = (
                    0 <= x <= self.window_width and
                    0 <= y <= self.detection_height
                )
                
                if in_detection_area and self.hidden:
                    self.root.after(0, self.show_window)
                elif not in_detection_area and not self.hidden:
                    # Add delay before hiding to prevent flickering
                    # time.sleep(0.3)
                    # Check again to confirm mouse is still not in area
                    x_check = self.root.winfo_pointerx()
                    y_check = self.root.winfo_pointery()
                    still_outside = not (
                        0 <= x_check <= self.window_width and
                        0 <= y_check <= self.detection_height
                    )
                    if still_outside:
                        self.root.after(0, self.hide_window)
                        
            except Exception as e:
                print(f"Error checking mouse position: {e}")
            # print("check")
            self.root.after(100, self.check_mouse_position)
            # time.sleep(0.1)  # Check every 100ms
            
    def show_window(self):
        """Show the seek bar window"""
        if self.hidden:
            self.root.attributes('-alpha', 1.0)
            self.hidden = False
            
    def hide_window(self):
        """Hide the seek bar window"""
        if not self.hidden:
            self.root.attributes('-alpha', 0.0)
            self.hidden = True
            
            

if __name__ == "__main__":
    root = tk.Tk()
    app = SeekBarApp(root)

    root.mainloop()