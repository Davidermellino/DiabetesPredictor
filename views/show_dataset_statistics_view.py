from tkinter import ttk

class ShowDatasetStatisticsView:
    def __init__(self, parent, statistics, missing_values):
        self.parent = parent
        self.statistics = statistics
        self.missing_values = missing_values
        
        self._create_widget()
        
    def _create_widget(self):
        label = ttk.Label(self.parent, text=f"Dataset Statistics", style="Title.TLabel")
        label.pack()
        
        
        num_record = self.statistics["records count"]
        
        for statistic, value in self.statistics.items():
            
            if statistic != "records count" and statistic != "features count":
                label = ttk.Label(self.parent, text=f"{statistic}: {value} ({value/num_record*100:.2f}%)", style="Title.TLabel")
                label.pack()
            else:
                label = ttk.Label(self.parent, text=f"{statistic}: {value}", style="Title.TLabel")
                label.pack()
                
        missing_value_label = ttk.Label(self.parent, text=f"Missing values: {self.missing_values}", style="Title.TLabel")
        missing_value_label.pack()
    
        