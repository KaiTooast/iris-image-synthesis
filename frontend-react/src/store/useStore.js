import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export const models = [
  { id: 'animagine_xl', name: 'Animagine XL 3.1', image: '/assets/thumbnails/thumbnail-animagine-xl-3.1.webp' },
  { id: 'anime_kawai', name: 'Anime Kawaii Diffusion', image: '/assets/thumbnails/thumbnail-anime-kawaii-diffusion.webp' },
  { id: 'anything_v5', name: 'Anything v5', image: '/assets/thumbnails/thumbnail-anything-v5.webp' },
  { id: 'aom3', name: 'AOM3', image: '/assets/thumbnails/thumbnail-aom3.webp' },
  { id: 'counterfeit_v3', name: 'Counterfeit v3.0', image: '/assets/thumbnails/thumbnail-counterfeit-v3.0.webp' },
  { id: 'dreamshaper_8', name: 'DreamShaper 8', image: '/assets/thumbnails/thumbnail-dreamshaper-8.webp' },
  { id: 'openjourney', name: 'Openjourney', image: '/assets/thumbnails/thumbnail-openjourney.webp' },
  { id: 'pixel_art', name: 'Pixel Art Diffusion', image: '/assets/thumbnails/thumbnail-pixel-art-diffusion.webp' },
  { id: 'stable_diffusion_2_1', name: 'Stable Diffusion 2.1', image: '/assets/thumbnails/thumbnail-sd-2.1.webp' },
  { id: 'stable_diffusion_3_5', name: 'SD 3.5 Medium', image: '/assets/thumbnails/thumbnail-sd-3.5-medium.webp' },
  { id: 'waifu_diffusion', name: 'Waifu Diffusion', image: '/assets/thumbnails/thumbnail-waifu-diffusion.webp' },
]

export const resolutions = [
  { value: '512x512', label: '1:1', sublabel: '512x512', icon: 'square' },
  { value: '512x768', label: '2:3', sublabel: '512×768', icon: 'portrait-narrow' },
  { value: '768x512', label: '3:2', sublabel: '768×512', icon: 'landscape-narrow' },
  { value: '768x1024', label: '3:4', sublabel: '768×1024', icon: 'portrait' },
  { value: '1024x768', label: '4:3', sublabel: '1024×768', icon: 'landscape' },
  { value: '720x1280', label: '9:16', sublabel: '720×1280', icon: 'portrait-tall' },
  { value: '1024x1024', label: 'HD', sublabel: '1024x1024', icon: 'square' },
  { value: 'custom', label: 'Custom', sublabel: 'manual', icon: 'custom' },
]

export const qualityPresets = {
  fast: { name: 'Fast (Draft)', desc: '15 steps • Quick preview', steps: 15, cfg: 7 },
  balanced: { name: 'Balanced', desc: '35 steps • Good quality', steps: 35, cfg: 10 },
  high: { name: 'High Quality', desc: '50 steps • Detailed output', steps: 50, cfg: 12 },
  extreme: { name: 'Extreme', desc: '100 steps • Maximum quality', steps: 100, cfg: 15 },
}

export const useStore = create(
  persist(
    (set) => ({
      settings: {
        model: 'anime_kawai',
        prompt: 'masterpiece, best quality, ultra-detailed, high resolution, cinematic lighting, 1girl, anime girl with cyan hair, cat ears, fox tail, wearing white tactical jacket, black pleated skirt, futuristic city background, soft bokeh, glowing eyes',
        negativePrompt: 'lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, jpeg artifacts, signature, watermark, blurry, deformed',
        resolution: '512x512',
        customWidth: 512,
        customHeight: 512,
        steps: 35,
        cfg: 10,
        seed: null,
        seedLocked: false,
        qualityPreset: 'balanced',
      },
      generation: {
        isGenerating: false,
        progress: 0,
        currentStep: 0,
        totalSteps: 0,
        status: 'Ready',
        currentImage: null,
        sessionImages: [],
      },
      setModel: (model) => set((state) => ({ settings: { ...state.settings, model } })),
      setPrompt: (prompt) => set((state) => ({ settings: { ...state.settings, prompt } })),
      setNegativePrompt: (negativePrompt) => set((state) => ({ settings: { ...state.settings, negativePrompt } })),
      setResolution: (resolution) => set((state) => ({ settings: { ...state.settings, resolution } })),
      setCustomDimensions: (width, height) => set((state) => ({ 
        settings: { ...state.settings, customWidth: width, customHeight: height } 
      })),
      setCustomWidth: (customWidth) => set((state) => ({ 
        settings: { ...state.settings, customWidth } 
      })),
      setCustomHeight: (customHeight) => set((state) => ({ 
        settings: { ...state.settings, customHeight } 
      })),
      setSteps: (steps) => set((state) => ({ settings: { ...state.settings, steps } })),
      setCfg: (cfg) => set((state) => ({ settings: { ...state.settings, cfg } })),
      setSeed: (seed) => set((state) => ({ settings: { ...state.settings, seed: seed || null } })),
      toggleSeedLock: () => set((state) => ({ 
        settings: { ...state.settings, seedLocked: !state.settings.seedLocked } 
      })),
      setQualityPreset: (preset) => {
        const config = qualityPresets[preset]
        set((state) => ({ 
          settings: { 
            ...state.settings, 
            qualityPreset: preset,
            steps: config.steps,
            cfg: config.cfg,
          } 
        }))
      },
      setGenerating: (isGenerating) => set((state) => ({ 
        generation: { ...state.generation, isGenerating } 
      })),
      setProgress: (progress, currentStep, totalSteps, status) => set((state) => ({ 
        generation: { ...state.generation, progress, currentStep, totalSteps, status } 
      })),
      setCurrentImage: (currentImage) => set((state) => ({ 
        generation: { ...state.generation, currentImage } 
      })),
      addSessionImage: (image) => set((state) => ({ 
        generation: { ...state.generation, sessionImages: [image, ...state.generation.sessionImages] } 
      })),
      randomizeSeed: () => set((state) => {
        if (state.settings.seedLocked) return state
        return { settings: { ...state.settings, seed: Math.floor(Math.random() * 999999) + 1 } }
      }),
    }),
    {
      name: 'iris-settings',
      partialize: (state) => ({ settings: state.settings }),
    }
  )
)
