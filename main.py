from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from dotenv import load_dotenv
import json
import asyncio
import logging
from google import genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('TravelAgentLive')

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Travel Agent AI - Live Chat")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize Google AI client
client = genai.Client(
    http_options={'api_version': 'v1alpha'}
)

MODEL = "gemini-2.0-flash-exp"

SYSTEM_PROMPT = """Eres un agente de viajes experto que ayuda a los usuarios a planificar sus viajes. 
Proporciona recomendaciones detalladas y personalizadas basadas en las preferencias del usuario.
Usa un tono conversacional y amigable."""

class TravelAgentLoop:
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.session = None
        self.config = {
            "generation_config": {
                "response_modalities": ["TEXT"]
            }
        }

    async def run(self):
        logger.debug('Connecting to model')
        try:
            async with client.aio.live.connect(model=MODEL, config=self.config) as session:
                self.session = session
                
                # Send initial greeting
                await self.send_message(SYSTEM_PROMPT)
                await self.send_message("Hola, soy tu asistente de viajes. ¿En qué puedo ayudarte?")
                await self.receive_response()

                while True:
                    try:
                        # Wait for message from client
                        data = await self.websocket.receive_text()
                        message = json.loads(data)["message"]
                        
                        # Send message and receive response
                        await self.send_message(message)
                        await self.receive_response()
                    except Exception as e:
                        logger.error(f"Error in message loop: {str(e)}")
                        break
        except Exception as e:
            logger.error(f"Error in session: {str(e)}")
            await self.websocket.send_json({
                "type": "error",
                "content": str(e)
            })

    async def send_message(self, text: str):
        logger.debug(f'Sending message: {text}')
        try:
            await self.session.send(text, end_of_turn=True)
            await self.websocket.send_json({
                "type": "status",
                "content": "Message sent"
            })
        except Exception as e:
            logger.error(f"Error sending message: {str(e)}")
            raise

    async def receive_response(self):
        try:
            turn = self.session.receive()
            async for chunk in turn:
                if chunk.text is not None:
                    await self.websocket.send_json({
                        "type": "text",
                        "content": chunk.text
                    })
        except Exception as e:
            logger.error(f"Error receiving response: {str(e)}")
            raise

@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    agent = TravelAgentLoop(websocket)
    
    try:
        await agent.run()
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Error in websocket connection: {str(e)}")
        try:
            await websocket.send_json({
                "type": "error",
                "content": str(e)
            })
        except:
            pass

@app.get("/")
async def root():
    return FileResponse('static/index.html')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
