#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dashboard_server.py — 多AGV仓库实时监控大屏（后端聚合 + 零依赖 HTTP 服务）

把分散的 ROS 话题聚合成一份 JSON 快照，并用 Python 标准库 http.server 提供：
    GET  /          → 大屏网页（HTML 内嵌在本文件，断网/无 CDN 也能用）
    GET  /state     → 当前状态 JSON（前端每 200ms 轮询一次）

订阅话题：
    /gazebo/model_states          各车世界坐标+朝向（gazebo 真值）
    /robot_i/carrying  (Int32)    各车载货 0=空 1=A 2=B 3=C
    /warehouse/shelf_goods (String "g0,..,g9")    货架库存
    /warehouse/shelf_reserved (String "r0,..,r9") 货架预约（推断各车目标）
    /warehouse/delivery (String "rid type")       每次投递触发 → 累计吞吐

用法：
    rosrun warehouse_qmix dashboard_server.py        # 默认端口 8080
    然后浏览器打开  http://localhost:8080
"""

import json
import math
import threading
import collections
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import rospy
from std_msgs.msg import String, Int32
from gazebo_msgs.msg import ModelStates

N_ROBOTS = 4
N_SHELVES = 10

# 货架 / 投递区在 Gazebo 世界坐标（来自 waypoints.xml 注释，单位 m）
SHELF_XY = [(-1.5, 6.5), (1.5, 6.5),
            (-8.5, 1.5), (-5.5, 1.5), (-8.5, -0.5), (-5.5, -0.5),
            (5.5, 1.5), (8.5, 1.5), (5.5, -0.5), (8.5, -0.5)]
DELIV_XY = {'A': (-2.5, -5.5), 'B': (-0.5, -5.5), 'C': (1.5, -5.5)}

# ── 共享状态（被 ROS 回调写、被 HTTP 线程读）──
_lock = threading.Lock()
_state = {
    'robots':   [{'id': i, 'x': 0.0, 'y': 0.0, 'yaw': 0.0,
                  'carrying': 0, 'target_shelf': -1} for i in range(N_ROBOTS)],
    'shelves':  [{'id': k, 'x': SHELF_XY[k][0], 'y': SHELF_XY[k][1],
                  'goods': 0, 'reserved': -1} for k in range(N_SHELVES)],
    'deliveries': {n: list(p) for n, p in DELIV_XY.items()},
    'cum_deliveries': 0,
    'by_type': {'A': 0, 'B': 0, 'C': 0},
    'rate_per_min': 0.0,
    'history': [],          # [[t_rel, cum], ...] 累计吞吐曲线
    't': 0.0,
}
_deliver_times = collections.deque(maxlen=2000)   # 每次投递的时刻，用于算速率
_t0 = None


# ─────────────────────────────────────────────────────────────────────────────
#  ROS 回调
# ─────────────────────────────────────────────────────────────────────────────
def _yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _model_states_cb(msg):
    with _lock:
        for i in range(N_ROBOTS):
            name = 'robot_{}'.format(i)
            if name in msg.name:
                idx = msg.name.index(name)
                p = msg.pose[idx]
                r = _state['robots'][i]
                r['x'] = round(p.position.x, 3)
                r['y'] = round(p.position.y, 3)
                r['yaw'] = round(_yaw_from_quat(p.orientation), 3)


def _carrying_cb(msg, rid):
    with _lock:
        _state['robots'][rid]['carrying'] = int(msg.data)


def _shelf_goods_cb(msg):
    try:
        goods = [int(x) for x in msg.data.strip().split(',')]
    except Exception:
        return
    with _lock:
        for k in range(min(N_SHELVES, len(goods))):
            _state['shelves'][k]['goods'] = goods[k]


def _reserved_cb(msg):
    try:
        rsv = [int(x) for x in msg.data.strip().split(',')]
    except Exception:
        return
    with _lock:
        for k in range(min(N_SHELVES, len(rsv))):
            _state['shelves'][k]['reserved'] = rsv[k]
        # 反推各车当前目标货架
        for r in _state['robots']:
            r['target_shelf'] = -1
        for k in range(min(N_SHELVES, len(rsv))):
            if 0 <= rsv[k] < N_ROBOTS:
                _state['robots'][rsv[k]]['target_shelf'] = k


def _delivery_cb(msg):
    try:
        parts = msg.data.strip().split()
        goods_type = int(parts[1])     # 1=A 2=B 3=C
    except Exception:
        return
    now = rospy.Time.now().to_sec()
    with _lock:
        _state['cum_deliveries'] += 1
        name = {1: 'A', 2: 'B', 3: 'C'}.get(goods_type)
        if name:
            _state['by_type'][name] += 1
    _deliver_times.append(now)


def _sampler(_evt):
    """每 2s 采样一次：更新速率与累计吞吐曲线。"""
    global _t0
    now = rospy.Time.now().to_sec()
    if _t0 is None:
        _t0 = now
    cutoff = now - 60.0
    while _deliver_times and _deliver_times[0] < cutoff:
        _deliver_times.popleft()
    with _lock:
        _state['rate_per_min'] = round(float(len(_deliver_times)), 1)
        _state['t'] = round(now - _t0, 1)
        _state['history'].append([round(now - _t0, 1), _state['cum_deliveries']])
        if len(_state['history']) > 180:        # 最多保留 6 分钟
            _state['history'] = _state['history'][-180:]


# ─────────────────────────────────────────────────────────────────────────────
#  HTTP 服务
# ─────────────────────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass   # 静音访问日志

    def do_GET(self):
        if self.path.startswith('/state'):
            with _lock:
                body = json.dumps(_state).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            body = HTML.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)


def main():
    rospy.init_node('dashboard_server')
    port = rospy.get_param('~port', 8080)

    rospy.Subscriber('/gazebo/model_states', ModelStates, _model_states_cb, queue_size=1)
    rospy.Subscriber('/warehouse/shelf_goods', String, _shelf_goods_cb)
    rospy.Subscriber('/warehouse/shelf_reserved', String, _reserved_cb)
    rospy.Subscriber('/warehouse/delivery', String, _delivery_cb)
    for i in range(N_ROBOTS):
        rospy.Subscriber('/robot_{}/carrying'.format(i), Int32,
                         lambda m, ii=i: _carrying_cb(m, ii))
    rospy.Timer(rospy.Duration(2.0), _sampler)

    httpd = ThreadingHTTPServer(('0.0.0.0', port), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    rospy.loginfo('[dashboard] 大屏已启动 → 浏览器打开 http://localhost:%d', port)
    rospy.spin()
    httpd.shutdown()


# ─────────────────────────────────────────────────────────────────────────────
#  前端大屏（纯 Canvas，无任何外部依赖）
# ─────────────────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>多AGV仓库调度 · 实时监控大屏</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:#eef3fb; color:#1e293b; font-family:"Microsoft YaHei",sans-serif;
         overflow:hidden; height:100vh; }
  #wrap { display:grid; grid-template-columns:1fr 380px; grid-template-rows:66px 1fr 150px;
          gap:14px; padding:14px; height:100vh; }
  #header { grid-column:1/3; display:flex; align-items:center; justify-content:space-between;
            background:#ffffff; border:1px solid #e3eaf3; border-radius:14px; padding:0 28px;
            box-shadow:0 2px 8px rgba(37,99,235,0.06); }
  #header h1 { font-size:22px; color:#1d4ed8; letter-spacing:2px; font-weight:700; }
  #header h1 small { color:#94a3b8; font-size:13px; letter-spacing:0; font-weight:400; }
  .kpis { display:flex; gap:38px; }
  .kpi { text-align:center; }
  .kpi .v { font-size:32px; font-weight:bold; color:#2563eb; line-height:1; }
  .kpi .v.blue { color:#2563eb; } .kpi .v.teal { color:#0ea5e9; }
  .kpi .l { font-size:12px; color:#94a3b8; margin-top:6px; }
  #mapbox { background:#ffffff; border:1px solid #e3eaf3; border-radius:14px; position:relative;
            box-shadow:0 2px 8px rgba(37,99,235,0.06); overflow:hidden; }
  #map { width:100%; height:100%; display:block; }
  #right { display:flex; flex-direction:column; gap:14px; min-height:0; }
  .panel { background:#ffffff; border:1px solid #e3eaf3; border-radius:14px; padding:14px;
           box-shadow:0 2px 8px rgba(37,99,235,0.06); }
  .panel h3 { font-size:14px; color:#1d4ed8; margin-bottom:12px; font-weight:600;
              border-left:4px solid #2563eb; padding-left:9px; }
  #shelfgrid { display:grid; grid-template-columns:repeat(5,1fr); gap:8px; }
  .shelf { aspect-ratio:1.4; border-radius:9px; display:flex; flex-direction:column;
           align-items:center; justify-content:center; font-size:11px; background:#f1f5f9;
           border:1px solid #e3eaf3; color:#94a3b8; transition:all .3s; }
  .shelf .g { font-size:18px; font-weight:bold; }
  #chartbox { flex:1; min-height:0; } #chart { width:100%; height:100%; display:block; }
  #cards { grid-column:1/3; display:grid; grid-template-columns:repeat(4,1fr); gap:14px; }
  .card { background:#ffffff; border:1px solid #e3eaf3; border-radius:14px; padding:13px 17px;
          border-left:5px solid #cbd5e1; box-shadow:0 2px 8px rgba(37,99,235,0.06); }
  .card .top { display:flex; justify-content:space-between; align-items:center; }
  .card .name { font-size:16px; font-weight:bold; }
  .card .badge { font-size:12px; padding:3px 12px; border-radius:11px; background:#eff6ff; color:#2563eb; }
  .card .row { font-size:13px; color:#64748b; margin-top:7px; }
  .card .row b { color:#1e293b; }
</style>
</head>
<body>
<div id="wrap">
  <div id="header">
    <h1>多AGV仓库智能调度系统 <small>QMIX 多智能体强化学习 · 实时监控</small></h1>
    <div class="kpis">
      <div class="kpi"><div class="v" id="kCum">0</div><div class="l">累计送达</div></div>
      <div class="kpi"><div class="v blue" id="kRate">0</div><div class="l">吞吐 (件/分)</div></div>
      <div class="kpi"><div class="v teal" id="kTime">0s</div><div class="l">运行时长</div></div>
    </div>
  </div>

  <div id="mapbox"><canvas id="map"></canvas></div>

  <div id="right">
    <div class="panel"><h3>货架库存</h3><div id="shelfgrid"></div></div>
    <div class="panel" id="chartbox" style="display:flex;flex-direction:column;">
      <h3>累计送达趋势</h3><canvas id="chart" style="flex:1"></canvas>
    </div>
  </div>

  <div id="cards"></div>
</div>

<script>
const RCOLOR = ['#e11d48','#2563eb','#16a34a','#ea580c'];   // 4 车配色（白底可读）
const GCOLOR = {0:'#cbd5e1',1:'#16a34a',2:'#d97706',3:'#0284c7'}; // 空/A绿/B橙/C蓝
const GNAME  = {0:'空',1:'A',2:'B',3:'C'};
// 世界坐标范围（含留白）
const WX0=-10, WX1=10, WY0=-8, WY1=8;

function fitCanvas(cv){ const r=cv.getBoundingClientRect(); const d=window.devicePixelRatio||1;
  cv.width=r.width*d; cv.height=r.height*d; const x=cv.getContext('2d'); x.setTransform(d,0,0,d,0,0); return x; }

function drawMap(ctx, cv, s){
  const W=cv.getBoundingClientRect().width, H=cv.getBoundingClientRect().height;
  ctx.clearRect(0,0,W,H);
  const sx=W/(WX1-WX0), sy=H/(WY1-WY0), sc=Math.min(sx,sy);
  const ox=(W-(WX1-WX0)*sc)/2, oy=(H-(WY1-WY0)*sc)/2;
  const X=x=>ox+(x-WX0)*sc, Y=y=>oy+(WY1-y)*sc;   // y 翻转：北在上

  // 网格
  ctx.strokeStyle='#eef3fb'; ctx.lineWidth=1;
  for(let gx=WX0;gx<=WX1;gx+=2){ ctx.beginPath(); ctx.moveTo(X(gx),Y(WY0)); ctx.lineTo(X(gx),Y(WY1)); ctx.stroke(); }
  for(let gy=WY0;gy<=WY1;gy+=2){ ctx.beginPath(); ctx.moveTo(X(WX0),Y(gy)); ctx.lineTo(X(WX1),Y(gy)); ctx.stroke(); }

  // 投递区
  for(const [n,p] of Object.entries(s.deliveries)){
    ctx.fillStyle='rgba(37,99,235,0.08)'; ctx.strokeStyle='#2563eb'; ctx.lineWidth=2;
    ctx.fillRect(X(p[0])-18,Y(p[1])-18,36,36); ctx.strokeRect(X(p[0])-18,Y(p[1])-18,36,36);
    ctx.fillStyle='#2563eb'; ctx.font='bold 14px sans-serif'; ctx.textAlign='center';
    ctx.fillText('投递'+n, X(p[0]), Y(p[1])+5);
  }
  // 货架
  for(const sh of s.shelves){
    const c=GCOLOR[sh.goods];
    ctx.fillStyle=sh.goods?c:'#f1f5f9'; ctx.strokeStyle=sh.goods?c:'#cbd5e1';
    ctx.lineWidth=2; ctx.fillRect(X(sh.x)-14,Y(sh.y)-14,28,28); ctx.strokeRect(X(sh.x)-14,Y(sh.y)-14,28,28);
    ctx.fillStyle=sh.goods?'#ffffff':'#94a3b8'; ctx.font='bold 13px sans-serif'; ctx.textAlign='center';
    ctx.fillText(sh.goods?GNAME[sh.goods]:('#'+sh.id), X(sh.x), Y(sh.y)+5);
  }
  // 机器人（带朝向三角 + 载货圈 + 到目标连线）
  for(const r of s.robots){
    const col=RCOLOR[r.id];
    if(r.target_shelf>=0){ const t=s.shelves[r.target_shelf];
      ctx.strokeStyle=col; ctx.globalAlpha=0.35; ctx.setLineDash([5,4]); ctx.lineWidth=1.5;
      ctx.beginPath(); ctx.moveTo(X(r.x),Y(r.y)); ctx.lineTo(X(t.x),Y(t.y)); ctx.stroke();
      ctx.setLineDash([]); ctx.globalAlpha=1; }
    const px=X(r.x), py=Y(r.y);
    ctx.save(); ctx.translate(px,py); ctx.rotate(-r.yaw);
    ctx.fillStyle=col; ctx.beginPath(); ctx.moveTo(13,0); ctx.lineTo(-9,-9); ctx.lineTo(-9,9); ctx.closePath(); ctx.fill();
    ctx.restore();
    if(r.carrying){ ctx.strokeStyle=GCOLOR[r.carrying]; ctx.lineWidth=3;
      ctx.beginPath(); ctx.arc(px,py,15,0,2*Math.PI); ctx.stroke(); }
    ctx.fillStyle='#1e293b'; ctx.font='bold 11px sans-serif'; ctx.textAlign='center';
    ctx.fillText('AGV'+r.id, px, py-19);
  }
}

function drawChart(ctx, cv, hist){
  const W=cv.getBoundingClientRect().width, H=cv.getBoundingClientRect().height;
  ctx.clearRect(0,0,W,H); const pad=28;
  if(hist.length<2) return;
  const tmax=hist[hist.length-1][0]||1, vmax=Math.max(5,hist[hist.length-1][1]);
  const X=t=>pad+(t/tmax)*(W-pad-6), Y=v=>H-pad-(v/vmax)*(H-pad-6);
  ctx.strokeStyle='#dbe4f0'; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(pad,6); ctx.lineTo(pad,H-pad); ctx.lineTo(W-6,H-pad); ctx.stroke();
  ctx.fillStyle='#94a3b8'; ctx.font='10px sans-serif'; ctx.textAlign='right';
  ctx.fillText(vmax, pad-4, Y(vmax)+4); ctx.fillText('0', pad-4, H-pad+4);
  // 蓝色面积填充
  ctx.beginPath(); hist.forEach((p,i)=>{ const xx=X(p[0]),yy=Y(p[1]); i?ctx.lineTo(xx,yy):ctx.moveTo(xx,yy); });
  ctx.lineTo(X(hist[hist.length-1][0]),H-pad); ctx.lineTo(X(hist[0][0]),H-pad); ctx.closePath();
  ctx.fillStyle='rgba(37,99,235,0.10)'; ctx.fill();
  ctx.strokeStyle='#2563eb'; ctx.lineWidth=2; ctx.beginPath();
  hist.forEach((p,i)=>{ const xx=X(p[0]),yy=Y(p[1]); i?ctx.lineTo(xx,yy):ctx.moveTo(xx,yy); }); ctx.stroke();
  const last=hist[hist.length-1]; ctx.fillStyle='#2563eb';
  ctx.beginPath(); ctx.arc(X(last[0]),Y(last[1]),3,0,2*Math.PI); ctx.fill();
}

function renderShelves(s){
  const g=document.getElementById('shelfgrid');
  if(g.children.length!==s.shelves.length){ g.innerHTML='';
    s.shelves.forEach(()=>{ const d=document.createElement('div'); d.className='shelf';
      d.innerHTML='<div class="g"></div><div class="id"></div>'; g.appendChild(d); }); }
  s.shelves.forEach((sh,i)=>{ const d=g.children[i];
    d.querySelector('.g').textContent=sh.goods?GNAME[sh.goods]:'—';
    d.querySelector('.id').textContent='货架'+sh.id;
    if(sh.goods){ d.style.background=GCOLOR[sh.goods]+'22'; d.style.borderColor=GCOLOR[sh.goods];
      d.style.color=GCOLOR[sh.goods]; } else { d.style.background='#f1f5f9';
      d.style.borderColor='#e3eaf3'; d.style.color='#94a3b8'; } });
}

function renderCards(s){
  const STATE=r=>{ if(r.carrying) return ['载货 '+GNAME[r.carrying]+' → 投递','#ea580c'];
    if(r.target_shelf>=0) return ['前往货架 '+r.target_shelf,'#2563eb']; return ['待命','#94a3b8']; };
  const c=document.getElementById('cards');
  if(c.children.length!==s.robots.length){ c.innerHTML='';
    s.robots.forEach(()=>{ const d=document.createElement('div'); d.className='card';
      d.innerHTML='<div class="top"><span class="name"></span><span class="badge"></span></div>'+
                  '<div class="row">载货：<b class="cg"></b></div>'+
                  '<div class="row">坐标：<b class="pos"></b></div>'; c.appendChild(d); }); }
  s.robots.forEach((r,i)=>{ const d=c.children[i], st=STATE(r);
    d.style.borderLeftColor=RCOLOR[r.id];
    d.querySelector('.name').textContent='AGV '+r.id; d.querySelector('.name').style.color=RCOLOR[r.id];
    const b=d.querySelector('.badge'); b.textContent=st[0]; b.style.color=st[1];
    d.querySelector('.cg').textContent=r.carrying?GNAME[r.carrying]:'空';
    d.querySelector('.pos').textContent='('+r.x.toFixed(1)+', '+r.y.toFixed(1)+')'; });
}

const mapCv=document.getElementById('map'), chartCv=document.getElementById('chart');
let mapCtx=fitCanvas(mapCv), chartCtx=fitCanvas(chartCv);
window.addEventListener('resize',()=>{ mapCtx=fitCanvas(mapCv); chartCtx=fitCanvas(chartCv); });

async function tick(){
  try{
    const s=await (await fetch('/state')).json();
    document.getElementById('kCum').textContent=s.cum_deliveries;
    document.getElementById('kRate').textContent=s.rate_per_min;
    document.getElementById('kTime').textContent=s.t+'s';
    drawMap(mapCtx,mapCv,s); drawChart(chartCtx,chartCv,s.history);
    renderShelves(s); renderCards(s);
  }catch(e){}
}
setInterval(tick,200); tick();
</script>
</body>
</html>
"""


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
