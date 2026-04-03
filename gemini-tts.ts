import { GoogleGenAI, Modality } from "@google/genai";

export class GeminiTTSPlayer {
  private audioContext: AudioContext | null = null;
  private isPlaying = false;
  private nextStartTime = 0;
  private apiKey: string;
  private currentSources: AudioBufferSourceNode[] = [];

  constructor(apiKey: string) {
    this.apiKey = apiKey;
  }

  private initAudioContext() {
    if (!this.audioContext) {
      this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)({
        sampleRate: 24000,
      });
    }
    if (this.audioContext.state === 'suspended') {
      this.audioContext.resume();
    }
  }

  async speak(text: string, voice: string = 'Zephyr', shouldStop: boolean = false) {
    if (!this.apiKey) {
      console.warn("TTS: No API key provided.");
      return;
    }

    this.initAudioContext();
    if (shouldStop) {
      this.stop();
    }
    this.isPlaying = true;
    
    // If not already playing, start from now. Otherwise, nextStartTime handles the queue.
    if (this.nextStartTime < this.audioContext!.currentTime) {
      this.nextStartTime = this.audioContext!.currentTime;
    }

    const ai = new GoogleGenAI({ apiKey: this.apiKey });
    
    try {
      // Using generateContentStream for low-latency if supported
      const response = await ai.models.generateContentStream({
        model: "gemini-2.5-flash-preview-tts",
        contents: [{ parts: [{ text }] }],
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: {
            voiceConfig: {
              prebuiltVoiceConfig: { voiceName: voice as any },
            },
          },
        },
      });

      for await (const chunk of response) {
        const base64Audio = chunk.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
        if (base64Audio) {
          this.enqueueAudio(base64Audio);
        }
      }
    } catch (error) {
      console.error("TTS Error:", error);
      this.isPlaying = false;
    }
  }

  private enqueueAudio(base64Data: string) {
    if (!this.audioContext) return;

    const binaryString = atob(base64Data);
    const len = binaryString.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }
    
    // PCM 16-bit Little Endian
    const int16Data = new Int16Array(bytes.buffer);
    const float32Data = new Float32Array(int16Data.length);
    for (let i = 0; i < int16Data.length; i++) {
      float32Data[i] = int16Data[i] / 32768.0;
    }

    const buffer = this.audioContext.createBuffer(1, float32Data.length, 24000);
    buffer.getChannelData(0).set(float32Data);

    const source = this.audioContext.createBufferSource();
    source.buffer = buffer;
    source.connect(this.audioContext.destination);
    
    const startTime = Math.max(this.audioContext.currentTime, this.nextStartTime);
    source.start(startTime);
    this.nextStartTime = startTime + buffer.duration;
    this.currentSources.push(source);
    
    source.onended = () => {
      const index = this.currentSources.indexOf(source);
      if (index > -1) {
        this.currentSources.splice(index, 1);
      }
      if (this.currentSources.length === 0) {
        this.isPlaying = false;
      }
    };
  }

  stop() {
    this.currentSources.forEach(source => {
      try {
        source.stop();
      } catch (e) {
        // Already stopped
      }
    });
    this.currentSources = [];
    this.isPlaying = false;
    this.nextStartTime = 0;
  }

  pause() {
    if (this.audioContext) {
      this.audioContext.suspend();
    }
  }

  resume() {
    if (this.audioContext) {
      this.audioContext.resume();
    }
  }

  setApiKey(apiKey: string) {
    this.apiKey = apiKey;
  }
}
