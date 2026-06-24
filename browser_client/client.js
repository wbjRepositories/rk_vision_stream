const video = document.getElementById("video");
const logBox = document.getElementById("log");
const connectBtn = document.getElementById("connectBtn");
const wsUrlInput = document.getElementById("wsUrl");

let ws = null;
let pc = null;
let remoteStream = null;
let answerSent = false;
let pendingLocalCandidates = [];
let pendingRemoteCandidates = [];

// 如果页面是通过 http://<RK3588-IP>/ 打开的，自动把 WebSocket 地址指向同一台板子；
// 如果直接 file:// 打开，则保留 HTML 里的默认值，用户可以手动改成板子的局域网 IP。
if (location.hostname) {
  wsUrlInput.value = `ws://${location.host}/ws`;
}

function log(msg) {
  console.log(msg);
  logBox.textContent += msg + "\n";
}

connectBtn.onclick = () => {
  const wsUrl = wsUrlInput.value;
  connectSignaling(wsUrl);
};

function connectSignaling(wsUrl) {
  log("connect websocket: " + wsUrl);

  if (ws) {
    ws.close();
  }
  if (pc) {
    pc.close();
    pc = null;
  }
  answerSent = false;
  pendingLocalCandidates = [];
  pendingRemoteCandidates = [];

  remoteStream = new MediaStream();
  video.srcObject = remoteStream;
  video.muted = true;
  video.playsInline = true;

  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    log("websocket connected");
  };

  ws.onerror = (event) => {
    log("websocket error");
    console.error(event);
  };

  ws.onclose = (event) => {
    log(`websocket closed code=${event.code} reason=${event.reason || ""} clean=${event.wasClean}`);
  };

  ws.onmessage = async (event) => {
    try {
      const msg = JSON.parse(event.data);
      log("recv: " + msg.type);

      if (msg.type === "offer") {
        await handleOffer(msg);
      } else if (msg.type === "ice") {
        await handleCandidate(msg);
      }
    } catch (err) {
      log("message handling error: " + err.message);
      console.error(err);
    }
  };
}

function createPeerConnection() {
  // 局域网查看只需要双方的 host candidate，不依赖公网 STUN/TURN。
  // 如果跨网段或公网访问，再在这里补充自己的 STUN/TURN 服务器。
  pc = new RTCPeerConnection({
    iceServers: []
  });

  pc.ontrack = (event) => {
    log(`ontrack kind=${event.track.kind} id=${event.track.id}`);

    if (!remoteStream) {
      remoteStream = new MediaStream();
      video.srcObject = remoteStream;
    }

    remoteStream.addTrack(event.track);

    event.track.onunmute = () => {
      log("track unmute");
      video.play().then(() => {
        log(`video play ok ${video.videoWidth}x${video.videoHeight}`);
      }).catch((err) => {
        log("video play failed: " + err.message);
      });
    };
    event.track.onmute = () => log("track mute");
    event.track.onended = () => log("track ended");

    video.play().catch((err) => {
      log("video play waiting: " + err.message);
    });
  };

  pc.onicecandidate = (event) => {
    if (event.candidate) {
      const candidateMessage = {
        type: "ice",
        candidate: event.candidate.candidate,
        sdpMid: event.candidate.sdpMid,
        sdpMLineIndex: event.candidate.sdpMLineIndex
      };

      if (answerSent) {
        sendIce(candidateMessage);
      } else {
        pendingLocalCandidates.push(candidateMessage);
      }
    }
  };

  pc.onconnectionstatechange = () => {
    log("connectionState: " + pc.connectionState);
  };

  pc.oniceconnectionstatechange = () => {
    log("iceConnectionState: " + pc.iceConnectionState);
  };

  pc.onicegatheringstatechange = () => {
    log("iceGatheringState: " + pc.iceGatheringState);
  };

  pc.onsignalingstatechange = () => {
    log("signalingState: " + pc.signalingState);
  };
}

function logSdpSummary(label, sdp) {
  const videoLine = sdp.split(/\r?\n/).find((line) => line.startsWith("m=video"));
  const h264Lines = sdp
    .split(/\r?\n/)
    .filter((line) => line.includes("H264") || line.includes("profile-level-id") || line.includes("packetization-mode"));
  log(`${label} ${videoLine || "no m=video"}`);
  h264Lines.forEach((line) => log(`${label} ${line}`));

  if (videoLine && videoLine.startsWith("m=video 0")) {
    log(`${label} video rejected by browser`);
  }
}

function sendIce(candidateMessage) {
  log("send candidate");

  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(candidateMessage));
  } else {
    log("skip candidate, websocket not open");
  }
}

async function handleOffer(msg) {
  if (!pc) {
    createPeerConnection();
  }

  log("set remote offer");
  logSdpSummary("offer", msg.sdp);

  await pc.setRemoteDescription({
    type: "offer",
    sdp: msg.sdp
  });
  for (const candidate of pendingRemoteCandidates) {
    await addRemoteCandidate(candidate);
  }
  pendingRemoteCandidates = [];

  log("create answer");

  const answer = await pc.createAnswer();

  await pc.setLocalDescription(answer);
  logSdpSummary("answer", pc.localDescription.sdp);

  log("send answer");

  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({
      type: "answer",
      sdp: pc.localDescription.sdp
    }));
    answerSent = true;
    pendingLocalCandidates.forEach(sendIce);
    pendingLocalCandidates = [];
  } else {
    log("cannot send answer, websocket not open");
  }
}

async function handleCandidate(msg) {
  if (!pc) {
    log("pc not ready, queue candidate");
    pendingRemoteCandidates.push(msg);
    return;
  }

  if (!pc.remoteDescription) {
    log("remote description not ready, queue candidate");
    pendingRemoteCandidates.push(msg);
    return;
  }

  await addRemoteCandidate(msg);
}

async function addRemoteCandidate(msg) {
  log("add remote candidate");

  await pc.addIceCandidate({
    candidate: msg.candidate,
    sdpMid: msg.sdpMid,
    sdpMLineIndex: msg.sdpMLineIndex
  });
}
