const themeKey='prl-theme';
const setTheme=t=>{document.documentElement.dataset.theme=t;localStorage.setItem(themeKey,t)};
setTheme(localStorage.getItem(themeKey)||'dark');
document.getElementById('themeToggle').onclick=()=>setTheme(document.documentElement.dataset.theme==='light'?'dark':'light');
fetch('/api/auth/me').then(async r=>{if(r.ok){const me=await r.json();if(me.authenticated)location.replace(me.dashboard)}}).catch(()=>{});
document.getElementById('loginForm').onsubmit=async e=>{e.preventDefault();const error=document.getElementById('loginError');const button=e.currentTarget.querySelector('button');button.disabled=true;error.textContent='';try{const r=await fetch('/api/auth/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({username:document.getElementById('username').value,password:document.getElementById('password').value})});const d=await r.json();if(!r.ok)throw new Error(d.detail||'Sign in failed');location.replace(d.dashboard)}catch(x){error.textContent=x.message}finally{button.disabled=false}};
