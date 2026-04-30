---
title: Proyecto Final IA aplicada
sidebar_position: 1
---

import proyectoFinal from "./img/Proyecto_PIA.pdf";
import imagenes from "./img/imagenes.zip"

A continuación se detalla el proyecto final de la UT5 IA Aplicada.
{/* Visor del PDF incrustado */}
<object data={proyectoFinal} type="application/pdf" width="100%" height="800px">
  <p>Tu navegador no soporta la visualización de PDFs. <a href={proyectoFinal} download>Descarga el proyecto haciendo clic aquí</a>.</p>
</object>
{/* Botón de descarga alternativa */}
<br/>
<div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap', marginTop: '16px' }}>

  <a href={proyectoFinal} download className="button button--primary">
    📄 Descargar PDF del proyecto
  </a>

  <a href={imagenes} download className="button button--secondary">
    🖼️ Descargar imágenes de prueba
  </a>

</div>