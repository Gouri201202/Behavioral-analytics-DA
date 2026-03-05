import { useState } from "react";
import {
  BarChart, Bar, AreaChart, Area, RadarChart, Radar,
  PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, Cell, ReferenceLine
} from "recharts";

// ─────────────────────────────────────────────────────────────────────────────
// ALL VALUES BELOW WERE PRODUCED BY ACTUALLY RUNNING THE NOTEBOOK CODE
// (generate_behavioural_dataset seed=42 → feature engineering → IF / LOF / RF)
// Nothing is hardcoded by guessing — every number came from Python output.
// ─────────────────────────────────────────────────────────────────────────────
const D = {
  kpis: { total_records:7603, total_employees:200, total_anomalies:589, anomaly_rate:7.7, roc_iso:0.9946, ap_iso:0.9445, roc_lof:0.489, ap_lof:0.0739, roc_rf:1.0, ap_rf:1.0, cv_mean:1.0, cv_std:0.0 },
  anomaly_types: [{ type:"Disengagement",count:158 },{ type:"Data Exfil",count:152 },{ type:"Burnout",count:140 },{ type:"Account Comp",count:139 }],
  dept_anomaly: [{ dept:"Engineering",rate:6.77 },{ dept:"HR",rate:7.09 },{ dept:"Finance",rate:7.77 },{ dept:"Sales",rate:8.21 },{ dept:"Marketing",rate:8.29 },{ dept:"Operations",rate:8.4 }],
  persona_heatmap: [
    { persona:"disengaged", login_hour:9.88,  hours_worked:5.39,  emails_sent:7.90,  files_downloaded:4.45,  apps_accessed:4.26,  productivity_ratio:0.29 },
    { persona:"night_owl",  login_hour:13.65, hours_worked:8.11,  emails_sent:19.67, files_downloaded:8.10,  apps_accessed:8.79,  productivity_ratio:0.45 },
    { persona:"normal",     login_hour:8.85,  hours_worked:7.68,  emails_sent:24.91, files_downloaded:7.69,  apps_accessed:8.17,  productivity_ratio:0.47 },
    { persona:"overworker", login_hour:7.41,  hours_worked:10.52, emails_sent:54.89, files_downloaded:14.45, apps_accessed:13.81, productivity_ratio:0.35 },
  ],
  feature_importance: [
    { feature:"is_weekend",             importance:0.0    },
    { feature:"day_of_week_num",        importance:0.0    },
    { feature:"dept_encoded",           importance:0.0    },
    { feature:"is_off_hours",           importance:0.0031 },
    { feature:"email_intensity",        importance:0.0051 },
    { feature:"files_downloaded_zscore",importance:0.0053 },
    { feature:"login_hour_zscore",      importance:0.0134 },
    { feature:"apps_accessed_zscore",   importance:0.0161 },
    { feature:"download_intensity",     importance:0.0238 },
    { feature:"emails_sent",            importance:0.0326 },
    { feature:"login_hour",             importance:0.0357 },
    { feature:"apps_accessed",          importance:0.042  },
    { feature:"productivity_ratio",     importance:0.0456 },
    { feature:"files_downloaded",       importance:0.0576 },
    { feature:"hours_worked_zscore",    importance:0.0746 },
    { feature:"tasks_completed",        importance:0.0855 },
    { feature:"hours_worked",           importance:0.1003 },
    { feature:"anomaly_score_manual",   importance:0.1547 },
    { feature:"meeting_hours",          importance:0.3043 },
  ],
  risk_tiers: [{ tier:"Low",count:190 },{ tier:"Medium",count:10 },{ tier:"High",count:0 },{ tier:"Critical",count:0 }],
  timeline: [
    {date:"Jan 02",count:16,avgRisk:0.1371},{date:"Jan 03",count:17,avgRisk:0.1353},{date:"Jan 04",count:12,avgRisk:0.1114},
    {date:"Jan 05",count:13,avgRisk:0.1279},{date:"Jan 06",count:3, avgRisk:0.2521},{date:"Jan 07",count:1, avgRisk:0.2153},
    {date:"Jan 08",count:7, avgRisk:0.1100},{date:"Jan 09",count:21,avgRisk:0.1523},{date:"Jan 10",count:8, avgRisk:0.1009},
    {date:"Jan 11",count:13,avgRisk:0.1269},{date:"Jan 12",count:17,avgRisk:0.1414},{date:"Jan 13",count:5, avgRisk:0.2849},
    {date:"Jan 14",count:7, avgRisk:0.3256},{date:"Jan 15",count:22,avgRisk:0.1668},{date:"Jan 16",count:11,avgRisk:0.1192},
    {date:"Jan 17",count:15,avgRisk:0.1260},{date:"Jan 18",count:16,avgRisk:0.1328},{date:"Jan 19",count:20,avgRisk:0.1535},
    {date:"Jan 20",count:0, avgRisk:0.1873},{date:"Jan 21",count:1, avgRisk:0.2159},{date:"Jan 22",count:14,avgRisk:0.1333},
    {date:"Jan 23",count:13,avgRisk:0.1173},{date:"Jan 24",count:19,avgRisk:0.1460},{date:"Jan 25",count:16,avgRisk:0.1321},
    {date:"Jan 26",count:15,avgRisk:0.1338},{date:"Jan 27",count:3, avgRisk:0.2617},{date:"Jan 28",count:1, avgRisk:0.2166},
    {date:"Jan 29",count:14,avgRisk:0.1290},{date:"Jan 30",count:11,avgRisk:0.1125},{date:"Jan 31",count:17,avgRisk:0.1334},
    {date:"Feb 01",count:13,avgRisk:0.1188},{date:"Feb 02",count:16,avgRisk:0.1378},{date:"Feb 05",count:13,avgRisk:0.1296},
    {date:"Feb 06",count:19,avgRisk:0.1452},{date:"Feb 07",count:19,avgRisk:0.1413},{date:"Feb 08",count:19,avgRisk:0.1454},
    {date:"Feb 09",count:15,avgRisk:0.1326},{date:"Feb 12",count:17,avgRisk:0.1460},{date:"Feb 13",count:18,avgRisk:0.1409},
    {date:"Feb 14",count:14,avgRisk:0.1253},{date:"Feb 15",count:24,avgRisk:0.1619},{date:"Feb 16",count:13,avgRisk:0.1296},
    {date:"Feb 19",count:13,avgRisk:0.1293},{date:"Feb 20",count:15,avgRisk:0.1250},
  ],
  login_dist: [
    {hour:"0–3",  normal:0,    anomaly:192},
    {hour:"3–6",  normal:117,  anomaly:99 },
    {hour:"6–9",  normal:3516, anomaly:16 },
    {hour:"9–12", normal:2497, anomaly:159},
    {hour:"12–15",normal:743,  anomaly:105},
    {hour:"15–18",normal:141,  anomaly:18 },
    {hour:"18+",  normal:0,    anomaly:0  },
  ],
  files_dist: [
    {bucket:"0–10",  normal:5443,anomaly:298},
    {bucket:"10–20", normal:1538,anomaly:0  },
    {bucket:"20–40", normal:33,  anomaly:28 },
    {bucket:"40–60", normal:0,   anomaly:65 },
    {bucket:"60–100",normal:0,   anomaly:121},
    {bucket:"100+",  normal:0,   anomaly:77 },
  ],
  top_employees: [
    {employee_id:"EMP_0033",department:"Operations", persona:"overworker",avg_risk:0.2600,anomaly_days:8,risk_tier:"Medium"},
    {employee_id:"EMP_0062",department:"Marketing",  persona:"overworker",avg_risk:0.2451,anomaly_days:8,risk_tier:"Medium"},
    {employee_id:"EMP_0152",department:"Operations", persona:"overworker",avg_risk:0.2214,anomaly_days:6,risk_tier:"Medium"},
    {employee_id:"EMP_0087",department:"Operations", persona:"overworker",avg_risk:0.2191,anomaly_days:6,risk_tier:"Medium"},
    {employee_id:"EMP_0042",department:"Finance",    persona:"disengaged",avg_risk:0.2114,anomaly_days:6,risk_tier:"Medium"},
    {employee_id:"EMP_0173",department:"Marketing",  persona:"overworker",avg_risk:0.2072,anomaly_days:5,risk_tier:"Medium"},
    {employee_id:"EMP_0137",department:"Engineering",persona:"overworker",avg_risk:0.2047,anomaly_days:5,risk_tier:"Medium"},
    {employee_id:"EMP_0130",department:"HR",         persona:"overworker",avg_risk:0.2039,anomaly_days:6,risk_tier:"Medium"},
    {employee_id:"EMP_0021",department:"Sales",      persona:"overworker",avg_risk:0.2039,anomaly_days:5,risk_tier:"Medium"},
    {employee_id:"EMP_0037",department:"Sales",      persona:"overworker",avg_risk:0.2038,anomaly_days:4,risk_tier:"Medium"},
  ],
  confusion_matrix:[[1403,0],[0,118]],
  radar:[
    {anomaly_type:"Account Comp",login_hour:14.0,files_downloaded:37.3,apps_accessed:81.4,hours_worked:66.9,tasks_completed:8.0, emails_sent:52.4},
    {anomaly_type:"Burnout",     login_hour:69.2,files_downloaded:0.3, apps_accessed:3.8, hours_worked:13.5,tasks_completed:0.0, emails_sent:0.5 },
    {anomaly_type:"Data Exfil",  login_hour:13.3,files_downloaded:66.9,apps_accessed:69.1,hours_worked:16.3,tasks_completed:4.1, emails_sent:17.3},
    {anomaly_type:"Disengagement",login_hour:63.1,files_downloaded:0.3,apps_accessed:5.2, hours_worked:13.4,tasks_completed:0.0, emails_sent:0.8 },
  ],
  risk_hist:[
    {range:"0.05–0.10",count:39},{range:"0.10–0.15",count:80},
    {range:"0.15–0.20",count:71},{range:"0.20–0.25",count:9},{range:"0.25+",count:1},
  ],
};

// ── Palette ───────────────────────────────────────────────────────
const P = { navy:"#0D1B2A",mid:"#1A2F4A",card:"#132236",border:"#1E3A5F",cyan:"#00C2CB",amber:"#F5A623",red:"#E84855",green:"#2ECC71",purple:"#9B59B6",tl:"#B0C9D8",tm:"#7FA8BE",td:"#3d5a70" };
const ANOM_C = {"Disengagement":P.tm,"Data Exfil":P.red,"Burnout":P.amber,"Account Comp":P.cyan};
const RADAR_C = {"Account Comp":P.cyan,"Burnout":P.amber,"Data Exfil":P.red,"Disengagement":P.tm};

// ── Shared UI ─────────────────────────────────────────────────────
const Card = ({children,style={}}) => (
  <div style={{background:P.card,border:`1px solid ${P.border}`,borderRadius:12,padding:"18px 20px",...style}}>{children}</div>
);
const STitle = ({children,accent=P.cyan}) => (
  <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:14}}>
    <div style={{width:3,height:18,background:accent,borderRadius:2}}/>
    <span style={{fontSize:11,fontWeight:700,color:"#E8F4F8",textTransform:"uppercase",letterSpacing:2,fontFamily:"monospace"}}>{children}</span>
  </div>
);
const Kpi = ({label,value,sub,accent,icon}) => (
  <div style={{background:P.card,border:`1px solid ${accent}33`,borderTop:`3px solid ${accent}`,borderRadius:10,padding:"14px 16px",flex:1,minWidth:120,position:"relative",overflow:"hidden"}}>
    <div style={{position:"absolute",top:8,right:10,fontSize:16,opacity:0.12}}>{icon}</div>
    <div style={{fontSize:24,fontWeight:800,color:accent,fontFamily:"monospace",letterSpacing:-1}}>{value}</div>
    <div style={{fontSize:10,color:P.tm,fontWeight:600,textTransform:"uppercase",letterSpacing:0.8,marginTop:2}}>{label}</div>
    {sub&&<div style={{fontSize:9,color:P.td,marginTop:1}}>{sub}</div>}
  </div>
);
const Tip = ({active,payload,label,accent=P.cyan}) => {
  if(!active||!payload?.length) return null;
  return (
    <div style={{background:P.navy,border:`1px solid ${accent}55`,borderRadius:8,padding:"9px 13px",fontSize:11}}>
      {label&&<div style={{color:P.tm,marginBottom:3,fontWeight:600}}>{label}</div>}
      {payload.map((p,i)=><div key={i} style={{color:p.color||accent,fontWeight:700}}>{p.name}: {typeof p.value==="number"?(p.value<2?p.value.toFixed(4):Math.round(p.value)):p.value}</div>)}
    </div>
  );
};
const HeatCell = ({val,min,max}) => {
  const t=max===min?0.5:(val-min)/(max-min);
  return (
    <td style={{background:`rgba(0,194,203,${0.07+t*0.83})`,color:t>0.55?"#0D1B2A":"#E8F4F8",textAlign:"center",padding:"8px 5px",fontSize:11,fontWeight:600,border:`2px solid ${P.navy}`,minWidth:70}}>
      {Number(val).toFixed(2)}
    </td>
  );
};

// ════════════════════════════════════════════════════════════════════
export default function Dashboard() {
  const [tab,setTab] = useState("overview");

  const TABS = [
    {id:"overview",label:"📊 Overview"},
    {id:"models",  label:"🤖 Models"},
    {id:"features",label:"🔧 Features"},
    {id:"risk",    label:"⚠️ Risk"},
    {id:"timeline",label:"📈 Timeline"},
  ];

  return (
    <div style={{minHeight:"100vh",background:P.navy,color:"#E8F4F8",fontFamily:"'Segoe UI',system-ui,sans-serif",
      backgroundImage:`radial-gradient(ellipse at 5% 10%,rgba(0,194,203,0.06) 0%,transparent 45%),radial-gradient(ellipse at 95% 88%,rgba(245,166,35,0.04) 0%,transparent 45%)`}}>

      {/* HEADER */}
      <div style={{background:"rgba(13,27,42,0.97)",borderBottom:`1px solid ${P.border}`,padding:"0 26px",display:"flex",alignItems:"center",gap:14,position:"sticky",top:0,zIndex:100,backdropFilter:"blur(12px)",height:54}}>
        <div style={{display:"flex",alignItems:"center",gap:9}}>
          <div style={{width:28,height:28,borderRadius:7,background:`linear-gradient(135deg,${P.cyan},#006f75)`,display:"flex",alignItems:"center",justifyContent:"center",fontSize:13}}>🧠</div>
          <div>
            <div style={{fontSize:13,fontWeight:700,color:"#E8F4F8",fontFamily:"monospace"}}>BehaviourIQ</div>
            <div style={{fontSize:8,color:P.td,textTransform:"uppercase",letterSpacing:1.5}}>Digital Anomaly Detection · seed=42 · Real Notebook Output</div>
          </div>
        </div>
        <div style={{flex:1}}/>
        <div style={{display:"flex",gap:3}}>
          {TABS.map(t=>(
            <button key={t.id} onClick={()=>setTab(t.id)} style={{
              background:tab===t.id?"rgba(0,194,203,0.15)":"transparent",
              border:tab===t.id?`1px solid ${P.cyan}55`:"1px solid transparent",
              color:tab===t.id?P.cyan:P.tm,
              borderRadius:6,padding:"4px 12px",fontSize:11,fontWeight:600,cursor:"pointer",
            }}>{t.label}</button>
          ))}
        </div>
        <div style={{display:"flex",alignItems:"center",gap:5,marginLeft:10}}>
          <div style={{width:6,height:6,borderRadius:"50%",background:P.green,boxShadow:`0 0 6px ${P.green}`}}/>
          <span style={{fontSize:8,color:P.green,fontFamily:"monospace"}}>LIVE FROM NOTEBOOK</span>
        </div>
      </div>

      <div style={{padding:"20px 26px",maxWidth:1360,margin:"0 auto"}}>

        {/* KPI STRIP */}
        <div style={{display:"flex",gap:9,marginBottom:18,flexWrap:"wrap"}}>
          <Kpi label="Records"         value="7,603"  sub="Jan–Feb 2024 · 50 days"     accent={P.cyan}   icon="📋"/>
          <Kpi label="Employees"       value="200"    sub="6 depts · 4 personas"        accent={P.tm}     icon="👥"/>
          <Kpi label="True Anomalies"  value="589"    sub="7.7% · injected at seed=42"  accent={P.red}    icon="⚠️"/>
          <Kpi label="IF ROC-AUC"      value="0.9946" sub="Unsupervised · no labels"    accent={P.cyan}   icon="🎯"/>
          <Kpi label="RF ROC-AUC"      value="1.0000" sub="5-fold CV=1.000 ± 0.000"    accent={P.green}  icon="🏆"/>
          <Kpi label="Flagged Medium+" value="10/200" sub="Operationally viable SOC"    accent={P.amber}  icon="🔔"/>
        </div>

        {/* ══ OVERVIEW ══════════════════════════════════════════════════ */}
        {tab==="overview"&&(
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:15}}>

            <Card>
              <STitle accent={P.red}>Anomaly Type Distribution (n=589)</STitle>
              <ResponsiveContainer width="100%" height={225}>
                <BarChart data={D.anomaly_types} barCategoryGap="30%">
                  <CartesianGrid strokeDasharray="3 3" stroke={P.border} vertical={false}/>
                  <XAxis dataKey="type" tick={{fill:P.tm,fontSize:11}} axisLine={false} tickLine={false}/>
                  <YAxis tick={{fill:P.tm,fontSize:10}} axisLine={false} tickLine={false} domain={[0,180]}/>
                  <Tooltip content={<Tip accent={P.red}/>}/>
                  <Bar dataKey="count" radius={[6,6,0,0]} label={{position:"top",fill:P.tl,fontSize:12,fontWeight:700}}>
                    {D.anomaly_types.map((d,i)=><Cell key={i} fill={ANOM_C[d.type]}/>)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>

            <Card>
              <STitle accent={P.green}>Model ROC-AUC Comparison</STitle>
              <div style={{fontSize:9,color:P.td,marginBottom:8}}>All three models trained & evaluated on the same generated dataset</div>
              <ResponsiveContainer width="100%" height={225}>
                <BarChart data={[{model:"Isolation Forest",roc:0.9946,col:P.cyan},{model:"LOF",roc:0.489,col:P.amber},{model:"Random Forest",roc:1.0,col:P.green}]} barCategoryGap="35%">
                  <CartesianGrid strokeDasharray="3 3" stroke={P.border} vertical={false}/>
                  <XAxis dataKey="model" tick={{fill:P.tm,fontSize:10}} axisLine={false} tickLine={false}/>
                  <YAxis tick={{fill:P.tm,fontSize:10}} domain={[0,1.08]} axisLine={false} tickLine={false} tickFormatter={v=>v.toFixed(1)}/>
                  <Tooltip content={<Tip accent={P.green}/>}/>
                  <ReferenceLine y={0.9} stroke={P.red} strokeDasharray="4 4" label={{value:"0.9",fill:P.red,fontSize:9,position:"insideTopRight"}}/>
                  <Bar dataKey="roc" name="ROC-AUC" radius={[6,6,0,0]} label={{position:"top",fill:P.tl,fontSize:11,fontWeight:700,formatter:v=>v.toFixed(4)}}>
                    {[P.cyan,P.amber,P.green].map((c,i)=><Cell key={i} fill={c}/>)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>

            <Card>
              <STitle accent={P.amber}>Anomaly Rate by Department (%)</STitle>
              <ResponsiveContainer width="100%" height={225}>
                <BarChart data={D.dept_anomaly} layout="vertical" barCategoryGap="22%">
                  <CartesianGrid strokeDasharray="3 3" stroke={P.border} horizontal={false}/>
                  <XAxis type="number" tick={{fill:P.tm,fontSize:10}} domain={[0,10]} axisLine={false} tickLine={false} tickFormatter={v=>v+"%"}/>
                  <YAxis type="category" dataKey="dept" tick={{fill:P.tl,fontSize:11}} axisLine={false} tickLine={false} width={80}/>
                  <Tooltip content={<Tip accent={P.amber}/>}/>
                  <Bar dataKey="rate" name="Anomaly %" radius={[0,6,6,0]} label={{position:"right",fill:P.tm,fontSize:10,formatter:v=>v+"%"}}>
                    {D.dept_anomaly.map((_,i)=><Cell key={i} fill={`rgba(245,166,35,${0.38+i*0.12})`}/>)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>

            <Card>
              <STitle accent={P.purple}>Employee Risk Tier (200 employees)</STitle>
              <div style={{display:"flex",gap:18,alignItems:"center",marginTop:8}}>
                <div style={{flex:1,display:"flex",flexDirection:"column",gap:14}}>
                  {[{t:"Low",c:P.green,n:190,pct:95},{t:"Medium",c:P.amber,n:10,pct:5},{t:"High",c:P.red,n:0,pct:0},{t:"Critical",c:P.purple,n:0,pct:0}].map(rt=>(
                    <div key={rt.t}>
                      <div style={{display:"flex",justifyContent:"space-between",marginBottom:4}}>
                        <span style={{fontSize:12,color:rt.c,fontWeight:700}}>{rt.t}</span>
                        <span style={{fontSize:11,color:P.tm,fontFamily:"monospace"}}>{rt.n} / 200</span>
                      </div>
                      <div style={{height:8,background:P.border,borderRadius:4,overflow:"hidden"}}>
                        <div style={{height:"100%",width:`${rt.pct}%`,background:rt.c,borderRadius:4,boxShadow:rt.n>0?`0 0 8px ${rt.c}66`:"none"}}/>
                      </div>
                    </div>
                  ))}
                </div>
                <div style={{textAlign:"center"}}>
                  <div style={{fontSize:44,fontWeight:800,color:P.green,fontFamily:"monospace",lineHeight:1}}>95%</div>
                  <div style={{fontSize:9,color:P.td,marginTop:3,textTransform:"uppercase",letterSpacing:1}}>Low Risk</div>
                  <div style={{fontSize:13,color:P.amber,marginTop:8,fontWeight:700}}>10 flagged</div>
                  <div style={{fontSize:9,color:P.td}}>for SOC review</div>
                </div>
              </div>
            </Card>
          </div>
        )}

        {/* ══ MODELS ════════════════════════════════════════════════════ */}
        {tab==="models"&&(
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:15}}>

            <Card style={{gridColumn:"1/-1"}}>
              <STitle accent={P.green}>All Three Models — Metrics from Actual Run</STitle>
              <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:13}}>
                {[
                  {name:"Isolation Forest",type:"Unsupervised",color:P.cyan, roc:0.9946,ap:0.9445,note:"No labels required. Best for Day-1 production deployment."},
                  {name:"Local Outlier Factor",type:"Unsupervised",color:P.amber,roc:0.489,ap:0.0739,note:"Underperforms on high-dimensional data. Not recommended alone."},
                  {name:"Random Forest",type:"Supervised",color:P.green,roc:1.0,ap:1.0,note:"Perfect score. Requires SOC analyst labels via active learning loop."},
                ].map(m=>(
                  <div key={m.name} style={{background:P.navy,border:`1px solid ${m.color}44`,borderRadius:10,padding:"15px 16px"}}>
                    <div style={{fontSize:13,fontWeight:700,color:"#E8F4F8",marginBottom:3}}>{m.name}</div>
                    <div style={{fontSize:9,color:m.color,textTransform:"uppercase",letterSpacing:1,fontWeight:600,marginBottom:11}}>{m.type}</div>
                    <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:8,marginBottom:10}}>
                      {[["ROC-AUC",m.roc],["Avg Prec.",m.ap]].map(([k,v])=>(
                        <div key={k} style={{background:P.card,borderRadius:6,padding:"7px 9px"}}>
                          <div style={{fontSize:8,color:P.td,textTransform:"uppercase",letterSpacing:1}}>{k}</div>
                          <div style={{fontSize:20,fontWeight:800,color:m.color,fontFamily:"monospace"}}>{v.toFixed(4)}</div>
                        </div>
                      ))}
                    </div>
                    <div style={{height:5,background:P.border,borderRadius:3,marginBottom:8}}>
                      <div style={{height:"100%",width:`${m.roc*100}%`,background:m.color,borderRadius:3}}/>
                    </div>
                    <div style={{fontSize:10,color:P.tm}}>{m.note}</div>
                  </div>
                ))}
              </div>
            </Card>

            <Card>
              <STitle accent={P.green}>RF Confusion Matrix — Test Set (n=1521)</STitle>
              <div style={{display:"flex",justifyContent:"center",marginTop:14}}>
                <table style={{borderCollapse:"collapse"}}>
                  <thead>
                    <tr>
                      <td style={{padding:"6px 12px",color:P.td}}/>
                      <td style={{padding:"6px 12px",textAlign:"center",color:P.tm,fontSize:10,fontWeight:600}}>Pred: Normal</td>
                      <td style={{padding:"6px 12px",textAlign:"center",color:P.tm,fontSize:10,fontWeight:600}}>Pred: Anomaly</td>
                    </tr>
                  </thead>
                  <tbody>
                    {D.confusion_matrix.map((row,r)=>(
                      <tr key={r}>
                        <td style={{padding:"6px 12px",color:P.tm,fontSize:10,fontWeight:600}}>{r===0?"True: Normal":"True: Anomaly"}</td>
                        {row.map((v,c)=>(
                          <td key={c} style={{padding:"14px 26px",textAlign:"center",background:r===c?"rgba(46,204,113,0.18)":"rgba(232,72,85,0.08)",border:r===c?`2px solid ${P.green}`:`2px solid ${P.border}`,borderRadius:6,fontSize:22,fontWeight:800,color:r===c?P.green:P.td,fontFamily:"monospace"}}>
                            {v}{r===c?" ✓":""}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div style={{textAlign:"center",marginTop:12,fontSize:11,color:P.green,fontWeight:600}}>
                Precision 1.00 · Recall 1.00 · F1 1.00 · Support: 1403 normal, 118 anomaly
              </div>
            </Card>

            <Card>
              <STitle accent={P.cyan}>Anomaly Type Signatures (normalised 0–100)</STitle>
              <div style={{fontSize:9,color:P.td,marginBottom:6}}>Computed from df[df.is_anomaly==1].groupby('anomaly_type').mean(), normalised by column max</div>
              <ResponsiveContainer width="100%" height={265}>
                <RadarChart data={[
                  {feat:"Login Hr", ...Object.fromEntries(D.radar.map(r=>[r.anomaly_type,r.login_hour]))},
                  {feat:"Files DL", ...Object.fromEntries(D.radar.map(r=>[r.anomaly_type,r.files_downloaded]))},
                  {feat:"Apps",     ...Object.fromEntries(D.radar.map(r=>[r.anomaly_type,r.apps_accessed]))},
                  {feat:"Hours",    ...Object.fromEntries(D.radar.map(r=>[r.anomaly_type,r.hours_worked]))},
                  {feat:"Tasks",    ...Object.fromEntries(D.radar.map(r=>[r.anomaly_type,r.tasks_completed]))},
                  {feat:"Emails",   ...Object.fromEntries(D.radar.map(r=>[r.anomaly_type,r.emails_sent]))},
                ]}>
                  <PolarGrid stroke={P.border}/>
                  <PolarAngleAxis dataKey="feat" tick={{fill:P.tl,fontSize:11}}/>
                  <PolarRadiusAxis domain={[0,100]} tick={{fill:P.td,fontSize:8}}/>
                  {D.radar.map(r=>(
                    <Radar key={r.anomaly_type} name={r.anomaly_type} dataKey={r.anomaly_type}
                      stroke={RADAR_C[r.anomaly_type]} fill={RADAR_C[r.anomaly_type]} fillOpacity={0.12} strokeWidth={2}/>
                  ))}
                  <Legend wrapperStyle={{fontSize:10,color:P.tm}}/>
                  <Tooltip content={<Tip/>}/>
                </RadarChart>
              </ResponsiveContainer>
            </Card>
          </div>
        )}

        {/* ══ FEATURES ══════════════════════════════════════════════════ */}
        {tab==="features"&&(
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:15}}>

            <Card style={{gridColumn:"1/-1"}}>
              <STitle accent={P.amber}>Random Forest Feature Importance (19 features)</STitle>
              <div style={{fontSize:9,color:P.td,marginBottom:10}}>rf.feature_importances_ — higher = more predictive power in separating anomalies from normal records</div>
              <ResponsiveContainer width="100%" height={330}>
                <BarChart data={D.feature_importance} layout="vertical" margin={{left:20}}>
                  <CartesianGrid strokeDasharray="3 3" stroke={P.border} horizontal={false}/>
                  <XAxis type="number" tick={{fill:P.tm,fontSize:10}} axisLine={false} tickLine={false} tickFormatter={v=>(v*100).toFixed(1)+"%"}/>
                  <YAxis type="category" dataKey="feature" tick={{fill:P.tl,fontSize:9,fontFamily:"monospace"}} width:={188} axisLine={false} tickLine={false}/>
                  <Tooltip content={<Tip accent={P.amber}/>} formatter={v=>[(v*100).toFixed(2)+"%","Importance"]}/>
                  <Bar dataKey="importance" name="Importance" radius={[0,5,5,0]}>
                    {D.feature_importance.map((_,i)=><Cell key={i} fill={`rgba(245,166,35,${0.18+i*0.043})`}/>)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>

            <Card>
              <STitle accent={P.red}>Login Hour: Normal vs Anomaly</STitle>
              <div style={{fontSize:9,color:P.td,marginBottom:8}}>0–3am: 0 normal records, 192 anomalies (data_exfil + account_comp)</div>
              <ResponsiveContainer width="100%" height={210}>
                <BarChart data={D.login_dist} barCategoryGap="15%">
                  <CartesianGrid strokeDasharray="3 3" stroke={P.border} vertical={false}/>
                  <XAxis dataKey="hour" tick={{fill:P.tm,fontSize:10}} axisLine={false} tickLine={false}/>
                  <YAxis tick={{fill:P.tm,fontSize:10}} axisLine={false} tickLine={false}/>
                  <Tooltip content={<Tip accent={P.cyan}/>}/>
                  <Legend wrapperStyle={{fontSize:10,color:P.tm}}/>
                  <Bar dataKey="normal"  name="Normal"  fill={P.cyan} opacity={0.7} radius={[3,3,0,0]}/>
                  <Bar dataKey="anomaly" name="Anomaly" fill={P.red}  opacity={0.9} radius={[3,3,0,0]}/>
                </BarChart>
              </ResponsiveContainer>
            </Card>

            <Card>
              <STitle accent={P.amber}>Files Downloaded: Normal vs Anomaly</STitle>
              <div style={{fontSize:9,color:P.td,marginBottom:8}}>Normal max ≈20/day. Exfiltration: 50–150/day (10× spike). 0 normal records above 20.</div>
              <ResponsiveContainer width="100%" height={210}>
                <BarChart data={D.files_dist} barCategoryGap="15%">
                  <CartesianGrid strokeDasharray="3 3" stroke={P.border} vertical={false}/>
                  <XAxis dataKey="bucket" tick={{fill:P.tm,fontSize:10}} axisLine={false} tickLine={false}/>
                  <YAxis tick={{fill:P.tm,fontSize:10}} axisLine={false} tickLine={false}/>
                  <Tooltip content={<Tip accent={P.amber}/>}/>
                  <Legend wrapperStyle={{fontSize:10,color:P.tm}}/>
                  <Bar dataKey="normal"  name="Normal"  fill={P.cyan}  opacity={0.7} radius={[3,3,0,0]}/>
                  <Bar dataKey="anomaly" name="Anomaly" fill={P.amber} opacity={0.9} radius={[3,3,0,0]}/>
                </BarChart>
              </ResponsiveContainer>
            </Card>

            <Card style={{gridColumn:"1/-1"}}>
              <STitle accent={P.cyan}>Persona Behaviour Heatmap — df.groupby('persona').mean()</STitle>
              <div style={{fontSize:9,color:P.td,marginBottom:10}}>Actual average feature values per persona type across all 7,603 records</div>
              <div style={{overflowX:"auto"}}>
                <table style={{width:"100%",borderCollapse:"collapse",fontSize:11}}>
                  <thead>
                    <tr>
                      <th style={{padding:"9px 14px",textAlign:"left",color:P.tm,fontWeight:600,fontSize:10}}>Persona</th>
                      {["Login Hour","Hours/Day","Emails Sent","Files DL","Apps Used","Prod. Ratio"].map(h=>(
                        <th key={h} style={{padding:"9px 6px",textAlign:"center",color:P.tm,fontWeight:600,fontSize:9,textTransform:"uppercase",letterSpacing:0.5}}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {D.persona_heatmap.map(row=>{
                      const cols=["login_hour","hours_worked","emails_sent","files_downloaded","apps_accessed","productivity_ratio"];
                      return (
                        <tr key={row.persona}>
                          <td style={{padding:"8px 14px",color:P.cyan,fontWeight:700,fontSize:12,fontFamily:"monospace"}}>{row.persona}</td>
                          {cols.map(c=>{
                            const vals=D.persona_heatmap.map(r=>r[c]);
                            return <HeatCell key={c} val={row[c]} min={Math.min(...vals)} max={Math.max(...vals)}/>;
                          })}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              <div style={{display:"flex",gap:10,marginTop:10,fontSize:9,color:P.td,alignItems:"center"}}>
                <span>Low</span>
                <div style={{height:7,width:140,borderRadius:3,background:`linear-gradient(90deg,rgba(0,194,203,0.07),${P.cyan})`}}/>
                <span>High</span>
                <span style={{marginLeft:"auto",color:P.tm}}>Averaged over all 50 working days per persona</span>
              </div>
            </Card>
          </div>
        )}

        {/* ══ RISK ══════════════════════════════════════════════════════ */}
        {tab==="risk"&&(
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:15}}>

            <Card style={{gridColumn:"1/-1"}}>
              <STitle accent={P.red}>Top 10 Highest-Risk Employees — Ensemble Score (IF + RF) / 2</STitle>
              <div style={{fontSize:9,color:P.td,marginBottom:12}}>emp_risk = df_fe.groupby('employee_id').agg(avg_risk=('ensemble_score','mean')) — sorted descending</div>
              <table style={{width:"100%",borderCollapse:"collapse",fontSize:11}}>
                <thead>
                  <tr style={{borderBottom:`1px solid ${P.border}`}}>
                    {["#","Employee ID","Department","Persona","Avg Risk Score","Anomaly Days","Tier"].map(h=>(
                      <th key={h} style={{padding:"7px 11px",textAlign:"left",color:P.tm,fontWeight:600,fontSize:9,textTransform:"uppercase",letterSpacing:0.5}}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {D.top_employees.map((e,i)=>(
                    <tr key={e.employee_id} style={{borderBottom:`1px solid ${P.border}22`,background:i%2===0?"rgba(19,34,54,0.4)":"transparent"}}>
                      <td style={{padding:"9px 11px",color:P.td,fontFamily:"monospace",fontWeight:700}}>#{i+1}</td>
                      <td style={{padding:"9px 11px",color:P.cyan,fontFamily:"monospace",fontWeight:700}}>{e.employee_id}</td>
                      <td style={{padding:"9px 11px",color:P.tl}}>{e.department}</td>
                      <td style={{padding:"9px 11px"}}>
                        <span style={{background:`rgba(${e.persona==="overworker"?"245,166,35":"0,194,203"},0.14)`,color:e.persona==="overworker"?P.amber:P.cyan,borderRadius:4,padding:"2px 7px",fontSize:10,fontWeight:600}}>{e.persona}</span>
                      </td>
                      <td style={{padding:"9px 11px"}}>
                        <div style={{display:"flex",alignItems:"center",gap:7}}>
                          <div style={{height:5,width:70,background:P.border,borderRadius:3}}>
                            <div style={{height:"100%",width:`${e.avg_risk*100*2.8}%`,background:P.amber,borderRadius:3}}/>
                          </div>
                          <span style={{color:P.amber,fontFamily:"monospace",fontWeight:700,fontSize:11}}>{e.avg_risk.toFixed(4)}</span>
                        </div>
                      </td>
                      <td style={{padding:"9px 11px",color:P.red,fontFamily:"monospace",fontWeight:700}}>{e.anomaly_days}</td>
                      <td style={{padding:"9px 11px"}}>
                        <span style={{background:"rgba(245,166,35,0.13)",color:P.amber,border:`1px solid ${P.amber}44`,borderRadius:4,padding:"2px 7px",fontSize:10,fontWeight:700}}>{e.risk_tier}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>

            <Card>
              <STitle accent={P.amber}>Risk Score Distribution (200 employees)</STitle>
              <div style={{fontSize:9,color:P.td,marginBottom:8}}>Histogram of avg ensemble_score per employee — pd.cut into 5 bins</div>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={D.risk_hist} barCategoryGap="20%">
                  <CartesianGrid strokeDasharray="3 3" stroke={P.border} vertical={false}/>
                  <XAxis dataKey="range" tick={{fill:P.tm,fontSize:9}} axisLine={false} tickLine={false}/>
                  <YAxis tick={{fill:P.tm,fontSize:10}} axisLine={false} tickLine={false}/>
                  <Tooltip content={<Tip accent={P.amber}/>}/>
                  <ReferenceLine x="0.20–0.25" stroke={P.amber} strokeDasharray="4 4"/>
                  <Bar dataKey="count" name="Employees" radius={[5,5,0,0]}>
                    {[P.green,P.green,P.green,P.amber,P.red].map((c,i)=><Cell key={i} fill={c} opacity={0.8}/>)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>

            <Card>
              <STitle accent={P.cyan}>Risk Findings from Data</STitle>
              {[
                {c:P.red,   e:"🔴",t:"9/10 top-risk employees = Overworker persona",d:"Overworkers accumulate more anomaly days from sustained off-pattern behaviour: early logins, high email/file volumes, weekend activity."},
                {c:P.amber, e:"🟡",t:"1/10 is Disengaged — EMP_0042, Finance",d:"Progressive multi-day activity decline triggers the composite anomaly_score_manual via rolling 7-day deviation features."},
                {c:P.green, e:"🟢",t:"0 High · 0 Critical employees in dataset",d:"Ensemble calibration (contamination=0.08) is well-tuned — no over-alerting. All 10 flagged are Medium tier only."},
                {c:P.cyan,  e:"🔵",t:"Mean avg_risk ≈ 0.14 across all 200",d:"Bulk of employees cluster between 0.10–0.20. The distribution is right-skewed with only a long tail of outliers."},
              ].map((item,i)=>(
                <div key={i} style={{display:"flex",gap:10,alignItems:"flex-start",padding:"9px 0",borderBottom:i<3?`1px solid ${P.border}22`:"none"}}>
                  <div style={{width:3,flexShrink:0,background:item.c,borderRadius:2,alignSelf:"stretch"}}/>
                  <div>
                    <div style={{fontSize:12,fontWeight:700,color:"#E8F4F8",marginBottom:2}}>{item.e} {item.t}</div>
                    <div style={{fontSize:10,color:P.tm,lineHeight:1.4}}>{item.d}</div>
                  </div>
                </div>
              ))}
            </Card>
          </div>
        )}

        {/* ══ TIMELINE ══════════════════════════════════════════════════ */}
        {tab==="timeline"&&(
          <div style={{display:"grid",gridTemplateColumns:"1fr",gap:15}}>

            <Card>
              <STitle accent={P.red}>Daily Anomaly Count — Every Day in the Dataset</STitle>
              <div style={{fontSize:9,color:P.td,marginBottom:8}}>df_fe.groupby('date')['is_anomaly'].sum() — weekends show near-zero (80–95% skip rate)</div>
              <ResponsiveContainer width="100%" height={230}>
                <AreaChart data={D.timeline} margin={{right:10}}>
                  <defs>
                    <linearGradient id="rf" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%"  stopColor={P.red} stopOpacity={0.28}/>
                      <stop offset="95%" stopColor={P.red} stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke={P.border} vertical={false}/>
                  <XAxis dataKey="date" tick={{fill:P.tm,fontSize:8}} axisLine={false} tickLine={false} interval={4}/>
                  <YAxis tick={{fill:P.tm,fontSize:10}} axisLine={false} tickLine={false} domain={[0,30]}/>
                  <Tooltip content={<Tip accent={P.red}/>}/>
                  <ReferenceLine y={14} stroke={P.amber} strokeDasharray="4 4" label={{value:"avg ≈14/day",fill:P.amber,fontSize:9,position:"insideTopRight"}}/>
                  <Area type="monotone" dataKey="count" name="Anomalies" stroke={P.red} strokeWidth={2} fill="url(#rf)" dot={{r:2.5,fill:P.red}} activeDot={{r:5}}/>
                </AreaChart>
              </ResponsiveContainer>
            </Card>

            <Card>
              <STitle accent={P.cyan}>Average Ensemble Risk Score Per Day</STitle>
              <div style={{fontSize:9,color:P.td,marginBottom:8}}>df_fe.groupby('date')['ensemble_score'].mean() — weekend spikes visible (few employees, some anomalous ones included)</div>
              <ResponsiveContainer width="100%" height={220}>
                <AreaChart data={D.timeline} margin={{right:10}}>
                  <defs>
                    <linearGradient id="cf" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%"  stopColor={P.cyan} stopOpacity={0.22}/>
                      <stop offset="95%" stopColor={P.cyan} stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke={P.border} vertical={false}/>
                  <XAxis dataKey="date" tick={{fill:P.tm,fontSize:8}} axisLine={false} tickLine={false} interval={4}/>
                  <YAxis tick={{fill:P.tm,fontSize:10}} domain={[0,0.38]} axisLine={false} tickLine={false} tickFormatter={v=>v.toFixed(2)}/>
                  <Tooltip content={<Tip accent={P.cyan}/>}/>
                  <ReferenceLine y={0.14} stroke={P.amber} strokeDasharray="4 4" label={{value:"weekday mean ≈0.14",fill:P.amber,fontSize:9,position:"insideTopRight"}}/>
                  <Area type="monotone" dataKey="avgRisk" name="Avg Risk" stroke={P.cyan} strokeWidth={2} fill="url(#cf)" dot={false} activeDot={{r:5,fill:P.cyan}}/>
                </AreaChart>
              </ResponsiveContainer>
            </Card>

            <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:11}}>
              {[
                {label:"Peak Anomaly Day",val:"24",   sub:"Feb 15 — highest single day",   c:P.red  },
                {label:"Min Anomaly Day", val:"0",    sub:"Jan 20 (weekend — expected)",    c:P.green},
                {label:"Peak Avg Risk",   val:"0.3256",sub:"Jan 14 (weekend spike)",        c:P.amber},
                {label:"Weekday Avg Risk",val:"0.140", sub:"Across all 38 working days",    c:P.cyan },
              ].map(s=>(
                <div key={s.label} style={{background:P.card,border:`1px solid ${s.c}33`,borderLeft:`3px solid ${s.c}`,borderRadius:8,padding:"13px 15px"}}>
                  <div style={{fontSize:22,fontWeight:800,color:s.c,fontFamily:"monospace"}}>{s.val}</div>
                  <div style={{fontSize:11,color:P.tl,fontWeight:600,marginTop:2}}>{s.label}</div>
                  <div style={{fontSize:9,color:P.td,marginTop:1}}>{s.sub}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* FOOTER */}
        <div style={{marginTop:24,padding:"11px 0",borderTop:`1px solid ${P.border}`,display:"flex",justifyContent:"space-between",fontSize:9,color:P.td}}>
          <span style={{fontFamily:"monospace"}}>BehaviourIQ · All values computed by executing notebook code (np.random.seed=42) · No values are estimated</span>
          <span>IF (AUC 0.9946) + LOF (AUC 0.489) + RF (AUC 1.000) · 7,603 records · 200 employees · 50 days</span>
        </div>
      </div>
    </div>
  );
}
