import tkinter as tk
import requests
import threading
import time

class SeekBarApp:
    def __init__(self, root=None, port=8123, get_info_fun=None, seek_fun=None):
        if not root: 
            self.root = tk.Tk()
        self.port = port
        self.root.overrideredirect(True)
        self.root.attributes('-topmost', True)
        
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        self.window_width = self.screen_width // 2
        self.window_height = 60
        self.root.geometry(f"{self.window_width}x{self.window_height}+0+0")
        
        self.root.attributes('-alpha', 0.0)  # Start hidden
        self.hidden = True
        self.detection_height = self.window_height+20  # Area height to detect cursor
        self.get_info_fun = get_info_fun
        self.seek_fun = seek_fun
        
        self.seek_var = tk.DoubleVar()
        self.seek_scale = tk.Scale(
            root,
            from_=0,
            to=1000,
            orient='horizontal',
            showvalue=False,
            # command=self.on_slider_interaction,
            sliderlength=20,
            width=self.window_height-10,
            troughcolor='#4d4d4d',
            bg='#333333',
            fg='white',
            variable=self.seek_var,
            highlightthickness=0
        )      
        self.seek_scale.pack(fill='x', padx=10, pady=5)

        def update(e):
            self.seek_scale.config( bg='#999999', troughcolor="#999999")
            self.root.after(500, lambda: self.seek_scale.config( bg='#333333',troughcolor="#4d4d4d"))
            def task():
                val = e.x / self.seek_scale.winfo_width() 
                print("event", val)
                self.on_slider_interaction(round(val*100))
            threading.Thread(target=task, daemon=True).start()

        self.seek_scale.bind("<Button-1>",  update)

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