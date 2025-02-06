def clear_content(frame):
    # Distrugge tutti i widget presenti nell'area di contenuto
    # per poter visualizzare un nuovo widget
    for widget in frame.winfo_children():
        widget.destroy()