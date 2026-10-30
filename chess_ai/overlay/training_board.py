"""Movable practice board overlay with optional Windows mouse-drag forwarding."""
from __future__ import annotations
import ctypes
import sys
import chess
from PyQt5 import QtCore, QtGui, QtWidgets

GLYPHS = {"K":"♔","Q":"♕","R":"♖","B":"♗","N":"♘","P":"♙",
          "k":"♚","q":"♛","r":"♜","b":"♝","n":"♞","p":"♟"}

def _forward_drag(start, end):
    """Send a desktop drag to the window underneath (Windows only)."""
    if sys.platform != "win32":
        return False
    class MI(ctypes.Structure):
        _fields_=[("dx",ctypes.c_long),("dy",ctypes.c_long),("data",ctypes.c_ulong),
                  ("flags",ctypes.c_ulong),("time",ctypes.c_ulong),("extra",ctypes.c_void_p)]
    class IU(ctypes.Union):
        _fields_=[("mi",MI)]
    class IN(ctypes.Structure):
        _fields_=[("type",ctypes.c_ulong),("u",IU)]
    u=ctypes.windll.user32
    vx,vy=u.GetSystemMetrics(76),u.GetSystemMetrics(77)
    vw,vh=max(1,u.GetSystemMetrics(78)),max(1,u.GetSystemMetrics(79))
    def xy(p):
        return (round((p.x()-vx)*65535/max(1,vw-1)),
                round((p.y()-vy)*65535/max(1,vh-1)))
    events=[]
    for p,f in ((start,0xC001),(start,0x0002),(end,0xC001),(end,0x0004)):
        x,y=xy(p); events.append(IN(0,IU(mi=MI(x,y,0,f,0,None))))
    arr=(IN*len(events))(*events)
    return u.SendInput(len(arr),arr,ctypes.sizeof(IN))==len(arr)

class BoardCanvas(QtWidgets.QWidget):
    def __init__(self, board, parent=None):
        super().__init__(parent); self.board=board; self.flipped=False
        self.forward=True; self.source=None; self.piece=None
        self.setMinimumSize(320,320)
    def square_at(self,p):
        s=min(self.width(),self.height()); x=(self.width()-s)/2; y=(self.height()-s)/2
        c=int((p.x()-x)/(s/8)); r=int((p.y()-y)/(s/8))
        if not(0<=c<8 and 0<=r<8): return None
        return chess.square(7-c if self.flipped else c, r if self.flipped else 7-r)
    def center(self,sq):
        s=min(self.width(),self.height()); x=(self.width()-s)/2; y=(self.height()-s)/2
        f,r=chess.square_file(sq),chess.square_rank(sq)
        c,row=(7-f,r) if self.flipped else (f,7-r)
        origin=self.mapToGlobal(QtCore.QPoint(0,0))
        return QtCore.QPoint(round(origin.x()+x+(c+.5)*s/8),round(origin.y()+y+(row+.5)*s/8))
    def mousePressEvent(self,e):
        if e.button()==QtCore.Qt.LeftButton:
            self.source=self.square_at(e.pos())
            self.piece=self.board.piece_at(self.source) if self.source is not None else None
            self.update()
    def mouseReleaseEvent(self,e):
        if self.source is None or self.piece is None: return
        a,b=self.source,self.square_at(e.pos()); piece=self.piece
        self.source=self.piece=None
        if b is None or a==b: self.update(); return
        self.board.remove_piece_at(a); self.board.remove_piece_at(b); self.board.set_piece_at(b,piece)
        self.board.turn=not self.board.turn; self.update()
        if self.forward:
            p1,p2=self.center(a),self.center(b); w=self.window(); w.hide()
            QtWidgets.QApplication.processEvents()
            if _forward_drag(p1,p2):
                QtCore.QTimer.singleShot(220,lambda:(w.show(),w.raise_()))
            else: w.show(); w.raise_()
    def paintEvent(self,e):
        p=QtGui.QPainter(self); p.setRenderHint(QtGui.QPainter.Antialiasing)
        s=min(self.width(),self.height()); x=(self.width()-s)/2; y=(self.height()-s)/2; cell=s/8
        for r in range(8):
            for c in range(8):
                rect=QtCore.QRectF(x+c*cell,y+r*cell,cell,cell)
                p.fillRect(rect,QtGui.QColor("#EEEED2" if (r+c)%2==0 else "#769656"))
                sq=self.square_at(QtCore.QPointF(x+(c+.5)*cell,y+(r+.5)*cell))
                pc=self.board.piece_at(sq) if sq is not None else None
                if pc and sq!=self.source:
                    f=QtGui.QFont("Segoe UI Symbol"); f.setPixelSize(int(cell*.78)); p.setFont(f)
                    p.setPen(QtGui.QColor("#202020" if pc.color else "#FAFAFA"))
                    p.drawText(rect,QtCore.Qt.AlignCenter,GLYPHS[pc.symbol()])
        if self.piece:
            f=QtGui.QFont("Segoe UI Symbol"); f.setPixelSize(int(cell*.78)); p.setFont(f)
            p.setPen(QtGui.QColor("#202020" if self.piece.color else "#FAFAFA"))
            p.drawText(self.rect(),QtCore.Qt.AlignCenter,GLYPHS[self.piece.symbol()])

class TrainingBoardOverlay(QtWidgets.QWidget):
    """Floating chessboard editor styled after a clean online board editor."""
    def __init__(self,parent=None):
        super().__init__(parent)
        self.setWindowTitle("Chess AI – Training Board")
        self.setWindowFlags(QtCore.Qt.Tool|QtCore.Qt.WindowStaysOnTopHint|QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground); self.resize(560,620)
        self.board=chess.Board(); root=QtWidgets.QVBoxLayout(self); root.setContentsMargins(8,8,8,8)
        head=QtWidgets.QHBoxLayout(); title=QtWidgets.QLabel("♞  TRAINING BOARD")
        title.setStyleSheet("color:white;font-weight:bold;"); head.addWidget(title); head.addStretch()
        close=QtWidgets.QPushButton("×"); close.clicked.connect(self.hide); head.addWidget(close); root.addLayout(head)
        self.canvas=BoardCanvas(self.board); root.addWidget(self.canvas,1)
        bar=QtWidgets.QHBoxLayout()
        for name,fn in (("Reset",self.reset_board),("Kings only",self.kings_only),("Rotate",self.rotate)):
            b=QtWidgets.QPushButton(name); b.clicked.connect(fn); bar.addWidget(b)
        for name, color in (("Turn white", chess.WHITE), ("Turn black", chess.BLACK)):
            turn_button=QtWidgets.QPushButton(name)
            turn_button.clicked.connect(lambda checked=False, side=color: self.set_turn(side))
            bar.addWidget(turn_button)
        self.forward_box=QtWidgets.QCheckBox("Forward drag"); self.forward_box.setChecked(True)
        self.forward_box.setStyleSheet("color:white"); self.forward_box.toggled.connect(self.set_forward); bar.addWidget(self.forward_box)
        root.addLayout(bar)
        opacity=QtWidgets.QHBoxLayout(); label=QtWidgets.QLabel("Transparency"); label.setStyleSheet("color:white"); opacity.addWidget(label)
        self.opacity=QtWidgets.QSlider(QtCore.Qt.Horizontal); self.opacity.setRange(35,100); self.opacity.setValue(92)
        self.opacity.valueChanged.connect(lambda v:self.setWindowOpacity(v/100)); opacity.addWidget(self.opacity,1); root.addLayout(opacity)
        self.setStyleSheet("QWidget{background:rgba(20,25,34,225);color:white} QPushButton{background:#29364b;color:white;border:1px solid #52627a;border-radius:5px;padding:5px}")
    def set_forward(self,v): self.canvas.forward=bool(v)
    def set_turn(self,side): self.board.turn=side; self.canvas.update()
    def reset_board(self): self.board.reset(); self.canvas.update()
    def kings_only(self):
        self.board.clear(); self.board.set_piece_at(chess.E1,chess.Piece(chess.KING,chess.WHITE))
        self.board.set_piece_at(chess.E8,chess.Piece(chess.KING,chess.BLACK)); self.canvas.update()
    def rotate(self): self.canvas.flipped=not self.canvas.flipped; self.canvas.update()
    def mousePressEvent(self,e):
        if e.button()==QtCore.Qt.LeftButton and e.pos().y()<42:
            self._drag_offset=e.globalPos()-self.frameGeometry().topLeft()
    def mouseMoveEvent(self,e):
        if hasattr(self,"_drag_offset") and e.buttons()&QtCore.Qt.LeftButton:
            self.move(e.globalPos()-self._drag_offset)
    def mouseReleaseEvent(self,e):
        if hasattr(self,"_drag_offset"): del self._drag_offset
